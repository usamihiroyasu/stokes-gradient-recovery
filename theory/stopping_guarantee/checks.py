"""Fixed algebra/AD/reference checks; no new solver-performance experiment."""
from pathlib import Path
from decimal import Decimal,localcontext
import argparse,datetime,hashlib,importlib.util,itertools,json,math,platform
import numpy as np
import scipy
from scipy.optimize import minimize
from certificate import geometry,tolerances,stopping_bound,target_thresholds
P=Path(__file__).resolve().parent;R=P.parents[1]
OUT=P/'results'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(x):
    if isinstance(x,np.ndarray):return pack(x.tolist())
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def save(name,x):(OUT/name).write_text(json.dumps(pack(x),indent=2,allow_nan=False)+'\n')


def inputs():
    q,_=np.linalg.qr(np.array([[1.,.2,-.1],[.3,1.,.2],[.1,-.2,1.]]))
    matrices=[np.diag([1.,.5,.5]),np.eye(3),q@np.diag([.5,1.5,3.])@q.T]
    cases=[]
    for mi,I,r,angle,mode in itertools.product(range(3),[.02,1.,4.],[0.,.25,.5,.75,.9,.99,.999,1.],[0.,.7],range(4)):
        e=np.array([np.cos(angle),np.sin(angle)]);phi,a,t,k=geometry(r);s=I*np.r_[1.,phi*e]
        n=np.r_[-1.,e]
        g=I*[np.array([.2,-.1,.3]),-.2*n,.2*n,1e-4*np.array([1.,2.,-1.])][mode]
        W=matrices[mi];z=s-np.linalg.solve(W,g)
        cases.append(dict(kind='quadratic',metric=mi,I=I,r=r,e=e,s=s,z=z,W=W,beta=0.))
    for I,r,angle in itertools.product([.02,1.,4.],[.25,.75,.99,1.],[0.,.7]):
        e=np.array([np.cos(angle),np.sin(angle)]);phi,a,t,k=geometry(r)
        cases.append(dict(kind='quartic',metric=0,I=I,r=r,e=e,s=I*np.r_[1.,phi*e],
                          z=np.array([1.,.3,-.2]),W=matrices[0],beta=.1))
    return cases


def objective(s,case):
    v=s-case['z'];W=case['W'];d=np.array([1.,.3,-.2]);u=float(d@v);beta=case['beta']
    return .5*float(v@W@v)+beta*u**4/4,W@v+beta*u**3*d,W+3*beta*u*u*np.outer(d,d)


def verify_cases():
    rows=[];reference=[];max_violation=max_uniform=0.;applicable=0
    for ci,case in enumerate(inputs()):
        s=case['s'];I=case['I'];r=case['r'];e=case['e'];phi,a,t,k=geometry(r)
        value,g,H=objective(s,case);mu=float(np.linalg.eigvalsh(case['W'])[0]);L=float(np.linalg.eigvalsh(H[1:,1:])[-1])
        b=float(g[0]+phi*(e@g[1:]));j=I*(a*np.outer(e,e)+t*(np.eye(2)-np.outer(e,e)))@g[1:]
        cr=I*I*a*a*float(e@H[1:,1:]@e)+I*k*float(e@g[1:])
        res=minimize(lambda z:objective(z,case)[0],np.array([1.5,0.,0.]),
            jac=lambda z:objective(z,case)[1],method='SLSQP',
            constraints=[dict(type='ineq',fun=lambda z:z[0]-np.linalg.norm(z[1:]),
                              jac=lambda z:np.r_[1.,-z[1:]/np.linalg.norm(z[1:])] if np.linalg.norm(z[1:])>1e-15 else np.array([1.,0.,0.]))],
            options=dict(ftol=1e-13,maxiter=1500))
        gap=value-float(res.fun)
        for error in [0.,1e-8,1e-4]:
            # Supplied values can partially cancel exact derivatives; inflate by valid budgets.
            bh=b+error*(-1 if ci%2 else 1);jh=j+error*np.array([.6,.8]);ch=cr+error
            eps,nu=tolerances(I,bh,jh,ch,error,error,error)
            bounds=stopping_bound(I,r,mu,L,eps,nu,Ibar=4.)
            max_violation=max(max_violation,gap-bounds['pointwise'])
            if np.isfinite(bounds['uniform']):applicable+=1;max_uniform=max(max_uniform,gap-bounds['uniform'])
            assert gap<=bounds['pointwise']+1e-8*(1+abs(value)),(ci,gap,bounds)
            if np.isfinite(bounds['uniform']):assert gap<=bounds['uniform']+1e-8*(1+abs(value))
            rows.append(dict(case=ci,kind=case['kind'],I=I,r=r,mu=mu,L=L,error=error,eps=eps,nu=nu,
                gap=gap,bounds=bounds,reference_success=bool(res.success),reference_status=int(res.status),
                reference_message=str(res.message),reference_feasibility=float(res.x[0]-np.linalg.norm(res.x[1:]))))
        reference.append(dict(case=ci,x=np.r_[np.log(I),r*e],z=case['z'],W=case['W'],beta=case['beta'],b=b,j=j,cr=cr))
    save('CASES.json',rows)
    np.savez_compressed(OUT/'AD_REFERENCE.npz',x=np.array([x['x'] for x in reference]),
        z=np.array([x['z'] for x in reference]),W=np.array([x['W'] for x in reference]),
        beta=np.array([x['beta'] for x in reference]),b=np.array([x['b'] for x in reference]),
        j=np.array([x['j'] for x in reference]),cr=np.array([x['cr'] for x in reference]))
    return dict(objectives=len(reference),input_error_checks=len(rows),uniform_applicable_checks=applicable,
        max_gap_above_pointwise_bound=max_violation,max_gap_above_uniform_bound=max_uniform,
        reference_nonacceptances=sum(not r['reference_success'] for r in rows[::3]),
        min_reference_feasibility=min(r['reference_feasibility'] for r in rows))


def asymptotics():
    rows=[]
    with localcontext() as context:
        context.prec=90;D=Decimal
        for k in range(1,13):
            delta=D(10)**(-k);r=1-delta;phi=2*r/(1+r*r);I=(1+phi)/(1+phi*phi)
            a=2*(1-r*r)/(1+r*r)**2;kappa=4*r*(r*r-3)/(1+r*r)**3
            m=(phi-1)/(1+phi*phi);eps=abs(a*m);gap=(1-phi)**2/(2*(1+phi*phi));curv=I*a*a+kappa*m
            assert curv>=0
            ratio=gap/(eps**(D(4)/3))
            rows.append(dict(family='sharp_first_tolerance',delta=str(delta),epsilon=str(eps),nu='0',gap=str(gap),ratio=str(ratio),curvature_over_I=str(curv)))
        for k in range(1,13):
            alpha=D(10)**(-k);rows.append(dict(family='sharp_curvature_tolerance',epsilon='0',nu=str(alpha),gap=str(alpha*alpha),ratio='1'))
        for k in range(1,9):
            r=D(10)**k;phi=2*r/(1+r*r);I=(1+phi/2)/(1+phi*phi);m=I*phi-D('.5')
            a=2*(1-r*r)/(1+r*r)**2;kap=4*r*(r*r-3)/(1+r*r)**3
            eps=abs(a*m);nu=max(D(0),-I*a*a-kap*m);gap=((I-1)**2+(I*phi-D('.5'))**2)/2
            rows.append(dict(family='unbounded_outer_sheet',r=str(r),epsilon=str(eps),nu=str(nu),gap=str(gap)))
        for k in range(1,17):
            I=D(10)**(-k);rawg=I*(I-1);rawh=I*(2*I-1);gap=(1-I)**2/2
            rows.append(dict(family='vanishing_intensity_raw_tests',I=str(I),raw_gradient_abs=str(abs(rawg)),raw_negative_hessian=str(max(D(0),-rawh)),normalized_b=str(I-1),gap=str(gap)))
    save('SHARPNESS_AND_COUNTEREXAMPLES.json',rows)
    first=[r for r in rows if r['family']=='sharp_first_tolerance']
    return dict(first_tolerance_limit=2**(-4/3),last_first_tolerance_ratio=float(first[-1]['ratio']),
        first_tolerance_cases=12,curvature_tolerance_cases=12,outer_sheet_cases=8,vanishing_intensity_cases=16)


def endpoint_audit():
    spec=importlib.util.spec_from_file_location('core',R/'verification/verify_warm_start.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    rows=json.loads((R/'application/real_data/RESULTS.json').read_text())['records']
    streams=json.loads((R/'application/real_data/INPUTS.json').read_text());ys={s['stream']:np.asarray(s['observations']) for s in streams}
    out=[]
    for row in rows:
        if row['arm']!='warm_start':continue
        x=np.array(row['final_x']);r=float(np.linalg.norm(x[1:]));changed=r>1
        if changed:x[1:]/=x[1:]@x[1:]
        s,J=c.geometry(x);r=float(np.linalg.norm(x[1:]));e=x[1:]/r if r else np.array([1.,0.]);y=ys[row['stream']][row['index']]
        head=c.fg(x,y)[1];direction=np.r_[0.,e];cr=float(direction@c.hessian(x,y)@direction)
        eps,nu=tolerances(s[0],head[0]/s[0],head[1:],cr)
        assert s[0]<=8 and r<=1+1e-14
        r=min(1.,r);B=stopping_bound(s[0],r,.5,.5,eps,nu,Ibar=8.)
        accepted=B['best']<=1e-6
        assert not accepted or not row['inaccurate']
        assert row['gap']<=B['best']+1e-10
        out.append(dict(dataset=row['dataset'],stratum=row['stratum'],method=row['method'],stream=row['stream'],index=row['index'],sheet_changed=changed,
            radius=r,I=s[0],eps=eps,nu=nu,bounds=B,original_gap=row['gap'],original_false_success=row['false_success'],accepted=accepted,
            accurate_rejected=not row['inaccurate'] and not accepted,physical_bound=row['physical_bound']))
    save('FROZEN_ENDPOINT_AUDIT.json',out)
    return dict(endpoints=len(out),sheet_changes=sum(r['sheet_changed'] for r in out),original_false_successes=sum(r['original_false_success'] for r in out),
        accepted=sum(r['accepted'] for r in out),false_accepts=sum(r['accepted'] and r['original_false_success'] for r in out),
        accurate_rejected=sum(r['accurate_rejected'] for r in out),uniform_accepted=sum(r['bounds']['uniform'] is not None and r['bounds']['uniform']<=1e-6 for r in pack(out)),
        new_solver_calls=0,full_head_hessian_queries=len(out),observations_changed=False,
        scope='Retrospective check of a theorem on already inspected endpoints; conditional float64 arithmetic, not validated rounding-error budgets or independent generalization.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True)
    paths=[P/'certificate.py',Path(__file__),R/'verification/verify_warm_start.py',R/'application/real_data/INPUTS.json',R/'application/real_data/RESULTS.json']
    hashes={str(f.relative_to(R)):sha(f) for f in paths}
    if args.register:
        assert not any(OUT.iterdir());save('PROTOCOL.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashes,
            scope='Theory validation, not a new optimizer comparison. Fixed 576 quadratics, 24 convex quartics, three derivative-error budgets, Decimal sharpness/counterexample families, and all 9284 existing warm endpoints. No adaptive grid expansion.',
            real_domain_Ibar=8.,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)));return
    assert not (OUT/'SUMMARY.json').exists();assert json.loads((OUT/'PROTOCOL.json').read_text())['source_sha256']==hashes
    result=dict(cases=verify_cases(),asymptotics=asymptotics(),frozen_endpoint_audit=endpoint_audit(),
                target_example=target_thresholds(.5,.5,8.,1e-6),source_sha256=hashes)
    save('SUMMARY.json',result);print(json.dumps(pack(result),indent=2))


if __name__=='__main__':main()
