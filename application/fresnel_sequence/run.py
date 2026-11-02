"""Fixed pilot: temporal Stokes estimation of a rocking dielectric interface.

Run from any directory. No network, training, or manuscript modification.
"""
from pathlib import Path
import collections,datetime,hashlib,importlib.util,itertools,json,platform,time,warnings
import numpy as np
import scipy
from scipy.optimize import minimize
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CORE=ROOT/'verification/verify_warm_start.py'
spec=importlib.util.spec_from_file_location('verified_stokes_core',CORE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,data):
    (HERE/name).write_text(json.dumps(c.pack(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def reflectance(theta,n):
    ci=np.cos(theta);ct=np.sqrt(1-(np.sin(theta)/n)**2)
    rs=((ci-n*ct)/(ci+n*ct))**2
    rp=((n*ci-ct)/(n*ci+ct))**2
    return rs,rp

def clean_sequence(mode,background,T=41):
    n=1.5;brewster=np.arctan(n);phase=2*np.pi*np.arange(T)/(T-1)
    theta=brewster+np.deg2rad(18)*np.cos(phase)
    alpha=np.zeros(T) if mode=='fixed' else .05*np.sin(phase)
    rs,rp=reflectance(theta,n);bs,bp=reflectance(brewster,n)
    scale=(bs+bp)/2
    intensity=(rs+rp)/(2*scale)+background
    pol=(rs-rp)/(2*scale)
    truth=np.stack([intensity,pol*np.cos(2*alpha),pol*np.sin(2*alpha)],axis=-1)
    return truth,theta,alpha

def generate():
    rows=[]
    for seq,(mode,background,photons,seed) in enumerate(itertools.product(['fixed','rocking'],[0.,.02,.1],[1000,10000],[0,1])):
        truth,theta,alpha=clean_sequence(mode,background)
        rng=np.random.default_rng(np.random.SeedSequence([20261022,seq,seed]))
        mean=photons*(truth@c.A.T)
        assert np.min(mean)>-1e-10
        electrons=rng.poisson(np.maximum(0,mean))+rng.normal(0,2.,mean.shape)
        y=electrons/photons
        rows.append(dict(sequence=seq,orientation_mode=mode,background=background,photon_scale=photons,seed=seed,truth=truth,incidence_radians=theta,AoLP_radians=alpha,observations=y))
    return rows

def mono_geometry(x):
    I=np.exp(x[0]);w=x[1:];d=1+w@w;f=w/np.sqrt(d)
    s=np.r_[I,I*f];J=np.eye(2)/np.sqrt(d)-np.outer(w,w)/d**1.5
    D=np.zeros((3,3));D[:,0]=s;D[1:,1:]=I*J
    return s,D

def mono_solve(x,y):
    calls=0;nonfinite=0
    def fun(x):
        nonlocal calls,nonfinite
        calls+=1;s,D=mono_geometry(x);v=c.A@s-y;g=D.T@c.A.T@v;f=.5*float(v@v)
        nonfinite+=int(not np.isfinite(f) or not np.all(np.isfinite(g)))
        return f,g
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');start=time.perf_counter()
        try:
            z=minimize(fun,x,jac=True,method='BFGS',options={'gtol':1e-6,'maxiter':500})
            out=dict(x=z.x,success=bool(z.success),status=int(z.status),message=str(z.message),nit=int(z.nit),loss=float(z.fun),native_gradient_inf=float(np.linalg.norm(z.jac,np.inf)))
        except Exception as e:
            out=dict(x=x,success=False,status=-999,message=repr(e),nit=0,loss=fun(x)[0],native_gradient_inf=None)
    out.update(counts={'function':calls,'gradient':calls,'hessian':0,'nonfinite_calls':nonfinite},elapsed_seconds=time.perf_counter()-start,warnings=dict(collections.Counter(str(v.message) for v in caught)))
    return out

def diagnose(x,y,monotone=False,guarded=False):
    s,D=mono_geometry(x) if monotone else c.geometry(x)
    g=c.A.T@(c.A@s-y);r=np.linalg.norm(x[1:]);e=x[1:]/r if r else np.array([1.,0.])
    near=not monotone and abs(r-1)<=1e-3
    B=c.certificate(s,g,r,e)
    curvature=None;trigger=False;rec_error=None;recB=None;nh=0
    if not monotone and (near or not guarded):
        z=c.diagnose(x,y);nh=1;curvature=z['normalized_radial_curvature'];trigger=z['trigger'];rec_error=z['recovered_gradient_error'];recB=z['recovered_bound']
    # Euclidean projection for an additional physical KKT residual.
    z=s-g;t=z[0];p=z[1:];rho=np.linalg.norm(p)
    projection=z if rho<=t else np.zeros(3) if rho<=-t else np.r_[(t+rho)/2,(t+rho)*p/(2*rho)]
    return dict(physical=s,radius=r,near_fold=near,normalized_radial_curvature=curvature,trigger=trigger,physical_bound=B,physical_accept=bool(np.isfinite(B) and B<=1e-6),physical_kkt_inf=float(np.max(np.abs(s-projection))),recovered_gradient_error=rec_error,recovered_bound=recB,audit_hessian_queries=nh)

def assess(out,y,minimum,monotone=False):
    s,_=mono_geometry(out['x']) if monotone else c.geometry(out['x'])
    v=c.A@s-y;loss=.5*float(v@v);gap=loss-minimum
    return dict(out,gap=gap,false_success=bool(out['success'] and (not np.isfinite(gap) or gap>1e-6)),inaccurate=bool(not np.isfinite(gap) or gap>1e-6),physical=s)

def preflight(inputs):
    rs,rp=reflectance(0,1.5);assert abs(rs-.04)<1e-15 and abs(rp-.04)<1e-15
    rs,rp=reflectance(np.arctan(1.5),1.5);assert rp<1e-28 and rs>0
    for row in inputs:
        s=np.array(row['truth']);assert np.min(s[:,0]-np.linalg.norm(s[:,1:],axis=1))>=-1e-13
    gradient_error=0.
    for x in [np.array([.2,.4,-.3]),np.array([-.5,3.,.7])]:
        y=np.array([.1,.4,.2,.3]);s,D=mono_geometry(x);g=D.T@c.A.T@(c.A@s-y)
        def loss(z):v=c.A@mono_geometry(z)[0]-y;return .5*v@v
        h=1e-5;fd=np.array([(loss(x+h*np.eye(3)[i])-loss(x-h*np.eye(3)[i]))/(2*h) for i in range(3)])
        gradient_error=max(gradient_error,float(np.max(np.abs(g-fd))))
    assert gradient_error<1e-7
    checks=[]
    for seq in [0,4,12,16]:
        for frame in [0,10,20,30,40]:
            y=np.array(inputs[seq]['observations'][frame]);s,minimum=c.cone_solution(y)
            def f(z):v=c.A@z-y;return .5*float(v@v),c.A.T@v
            z=minimize(f,np.array([1.,0.,0.]),jac=True,method='SLSQP',constraints=[{'type':'ineq','fun':lambda z:z[0]-np.linalg.norm(z[1:])}],options={'ftol':1e-13,'maxiter':1000})
            err=abs(z.fun-minimum);assert z.success and err<1e-9
            checks.append(dict(sequence=seq,frame=frame,success=bool(z.success),loss_difference=err))
    return dict(monotone_gradient_finite_difference_max_error=gradient_error,cone_SLSQP_checks=checks)

def summarize(records):
    groups=collections.defaultdict(list)
    for r in records:groups[(r['method'],r['arm'])].append(r)
    out=[]
    for (method,arm),rows in groups.items():
        vals=[r['final'] for r in rows];diags=[r['final_diagnostic'] for r in rows]
        out.append(dict(method=method,arm=arm,frames=len(rows),false_success=sum(v['false_success'] for v in vals),solver_nonacceptances=sum(not v['success'] for v in vals),inaccurate=sum(v['inaccurate'] for v in vals),interventions=sum(r['intervene'] for r in rows),max_gap=max(v['gap'] for v in vals),native_function_gradient_pairs=sum(r['initial']['counts']['function'] for r in rows),restart_function_gradient_pairs=sum(r['restart']['counts']['function']+3 for r in rows if r['restart'] is not None),audit_hessian_queries=sum(r['initial_diagnostic']['audit_hessian_queries'] for r in rows),physical_false_accept=sum(d['physical_accept'] and v['inaccurate'] for d,v in zip(diags,vals)),stokes_RMSE=float(np.sqrt(np.mean([np.mean((np.array(v['physical'])-np.array(r['truth']))**2) for r,v in zip(rows,vals)]))),optimizer_seconds=sum(r['initial']['elapsed_seconds']+(r['restart']['elapsed_seconds'] if r['restart'] is not None else 0) for r in rows)))
    return out

def main():
    protocol={
        'registered_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'One predeclared local Fresnel sequence pilot; no parameter search, network training, paper edits or dataset download',
        'physics':'Air to nonabsorbing n=1.5 interface; unpolarized incident light; single-interface Fresnel reflection; normalized specular S0=1 at Brewster; constant unpolarized background 0, 0.02 or 0.1 in the same units',
        'incidence':'atan(1.5) + 18 degrees*cos(2*pi*t/40), t=0..40',
        'AoLP':'fixed zero, or 0.05*sin(2*pi*t/40) radians; two physical motion controls',
        'camera':'Ideal analyzer angles 0,45,90,135 degrees; independent Poisson(photon_scale*A*s) plus Gaussian read noise SD 2 electrons; divide by photon_scale; no clipping or denoising',
        'photon_scale':[1000,10000],'seed':[0,1],'seed_rule':'SeedSequence([20261022, configuration_index, seed]); independent configurations and frames',
        'unique_sequences':24,'frames_per_sequence':41,'unique_observations':984,
        'fit':'Unweighted least squares, not claimed Poisson maximum likelihood',
        'arms':['BFGS and L-BFGS-B: folded baseline','BFGS and L-BFGS-B: radius-guarded curvature restart','BFGS and L-BFGS-B: physical gap audit plus same radial restart','BFGS: monotone ball-map baseline','direct weighted cone reference every frame'],
        'sequence_state':'Each solver/arm carries its own previous raw estimate; all begin at [0,0.2,-0.1], independently of truth; retain failures',
        'native_gtol':1e-6,'objective_gap_threshold':1e-6,'maxiter':500,
        'restart':'At most once per frame; choose least-loss candidate among radial offsets -0.01,+0.01,0, then same solver; interventions propagate only in their own arm',
        'curvature_gate':'abs(r-1)<=1e-3 and c_r/I^2 < -1e-7; same frozen gate as manuscript',
        'report':'All native flags, physical objective gaps, truth Stokes RMSE, independent physical certificate, evaluation counts and warnings; raw endpoint diagnostic Hessians are post-hoc coverage overhead; no universal speed claim',
        'stop_rule':'No adaptation to outcome. Zero occurrence or absence of added diagnostic value is retained; do not add to manuscript automatically.',
        'physics_source':'https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission',
        'code_sha256':digest(Path(__file__)),'core_sha256':digest(CORE),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'host':platform.node()}
    }
    assert not (HERE/'PROTOCOL.json').exists(),'Use a fresh directory; preserve registered pilot.'
    save('PROTOCOL.json',protocol);inputs=generate();save('inputs.json',inputs)
    save('preflight.json',preflight(inputs));records=[];start=time.time();cone=[]
    for inp in inputs:
        xs={(method,arm):np.array([0.,.2,-.1]) for method in ['BFGS','L-BFGS-B'] for arm in ['baseline','curvature_gate','physical_audit']};xs[('BFGS','monotone')]=np.array([0.,.2,-.1])
        for frame,(y,truth) in enumerate(zip(inp['observations'],inp['truth'])):
            opt,minimum=c.cone_solution(y);cone.append(dict(sequence=inp['sequence'],frame=frame,physical=opt,minimum=minimum,truth=truth))
            for (method,arm),x in xs.items():
                mono=arm=='monotone'
                initial=assess(mono_solve(x,y) if mono else c.solve(x,y,method,1e-6),y,minimum,mono)
                diag=diagnose(initial['x'],y,mono,guarded=arm!='baseline')
                intervene=(arm=='curvature_gate' and diag['trigger']) or (arm=='physical_audit' and not diag['physical_accept'])
                again=assess(c.restart(initial,y,method,1e-6),y,minimum) if intervene else None
                final=again if again is not None else initial
                final_diag=diagnose(final['x'],y,mono,guarded=True)
                xs[(method,arm)]=np.array(final['x'])
                records.append(dict(sequence=inp['sequence'],orientation_mode=inp['orientation_mode'],background=inp['background'],photon_scale=inp['photon_scale'],seed=inp['seed'],frame=frame,truth=truth,method=method,arm=arm,initial=initial,initial_diagnostic=diag,intervene=bool(intervene),restart=again,final=final,final_diagnostic=final_diag))
        print(json.dumps({'completed_sequences':inp['sequence']+1,'of':len(inputs),'elapsed_seconds':round(time.time()-start,2)}),flush=True)
    save('results.json',{'records':records,'summary':summarize(records)})
    save('cone_reference.json',cone)
    cone_rmse=float(np.sqrt(np.mean([(np.array(r['physical'])-np.array(r['truth']))**2 for r in cone])))
    baseline=[r for r in records if r['arm']=='baseline']
    coverage=[]
    for method in ['BFGS','L-BFGS-B']:
        rows=[r for r in baseline if r['method']==method]
        coverage.append(dict(method=method,baseline_false_success=sum(r['final']['false_success'] for r in rows),false_success_detected_by_curvature=sum(r['final']['false_success'] and r['initial_diagnostic']['trigger'] for r in rows),accurate_success_triggered=sum(r['final']['success'] and not r['final']['inaccurate'] and r['initial_diagnostic']['trigger'] for r in rows),false_success_rejected_by_physical=sum(r['final']['false_success'] and not r['initial_diagnostic']['physical_accept'] for r in rows),near_fold_endpoints=sum(r['initial_diagnostic']['near_fold'] for r in rows)))
    summary={'protocol_sha256':digest(HERE/'PROTOCOL.json'),'inputs_sha256':digest(HERE/'inputs.json'),'results_sha256':digest(HERE/'results.json'),'elapsed_seconds':time.time()-start,'unique_observations':len(cone),'optimizer_frame_fits':len(records),'direct_cone_stokes_RMSE':cone_rmse,'arms':summarize(records),'baseline_diagnostic_coverage':coverage,'no_adaptive_runs':True,'manuscript_changed':False}
    save('SUMMARY.json',summary);print(json.dumps(c.pack(summary)),flush=True)

if __name__=='__main__':main()
