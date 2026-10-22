"""Bounded verification in independent Cartesian coordinates. Run from this directory.

Original strategy modules are deliberately not imported. All source inputs are read-only.
"""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, platform, socket, sys, time, warnings
import numpy as np
import scipy
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
A = np.array([[.5,.5,0],[.5,0,.5],[.5,-.5,0],[.5,0,-.5]])
W = np.diag([1.,.5,.5])
M = np.diag([1.,2.,2.])
GAP = 1e-6

def pack(o):
    if isinstance(o, np.ndarray): return [pack(v) for v in o.tolist()]
    if isinstance(o, dict): return {str(k):pack(v) for k,v in o.items()}
    if isinstance(o, (list,tuple)): return [pack(v) for v in o]
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float,np.floating)): return float(o) if np.isfinite(o) else None
    return o

def save(name,data):
    (HERE/name).write_text(json.dumps(pack(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def geometry(x, second=False):
    x=np.asarray(x,dtype=np.float64); intensity=np.exp(x[0]); w=x[1:]; d=1+sum(w*w)
    f=2*w/d; s=np.r_[intensity,intensity*f]
    J=np.zeros((2,2)); eye=np.eye(2)
    for i in range(2):
        for j in range(2): J[i,j]=2*eye[i,j]/d-4*w[i]*w[j]/d**2
    D=np.zeros((3,3));D[:,0]=s;D[1:,1:]=intensity*J
    if not second: return s,D
    T=np.zeros((3,3,3));T[:,0,0]=s
    T[1:,0,1:]=intensity*J;T[1:,1:,0]=intensity*J
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T[i+1,j+1,k+1]=intensity*(-4*(eye[i,j]*w[k]+eye[i,k]*w[j]+eye[j,k]*w[i])/d**2+16*w[i]*w[j]*w[k]/d**3)
    return s,D,T

def fg(x,y):
    s,D=geometry(x);v=A@s-y;g=A.T@v
    return .5*float(v@v),D.T@g

def hessian(x,y):
    s,D,T=geometry(x,True);g=A.T@(A@s-y)
    return D.T@W@D+np.einsum('i,ijk->jk',g,T)

def cone_solution(y):
    # Analytic scalar reduction: minimize over I>=0 and a parallel p, with |p|<=I.
    z=np.array([sum(y)/2, y[0]-y[2],y[1]-y[3]])
    rho=np.hypot(z[1],z[2])
    if z[0]>=rho: s=z
    elif 2*z[0]+rho<=0: s=np.zeros(3)
    else:
        intensity=(2*z[0]+rho)/3;s=np.r_[intensity,intensity*z[1:]/rho]
    v=A@s-y
    return s,.5*float(v@v)

def certificate(s,g,r,e):
    if r==0: return .5*float(g@M@g)
    n=np.r_[-1.,e];sigma=max(0.,float(s[0]-e@s[1:]))
    lam=max(0.,-float(n@M@g+sigma)/float(n@M@n));v=g+lam*n
    return .5*float(v@M@v)+lam*sigma

def diagnose(x,y):
    s,D=geometry(x);g=A.T@(A@s-y);head=D.T@g;r=float(np.linalg.norm(x[1:]));e=x[1:]/r if r else np.array([1.,0.])
    direction=np.r_[0.,e];H=hessian(x,y);cr=float(direction@H@direction)
    B=certificate(s,g,r,e);near=abs(r-1)<=1e-3;trigger=near and cr/s[0]**2 < -1e-7
    rec=None;recB=None
    if near:
        d=1+r*r;ar=2*(1-r*r)/d**2;tr=2/d;kappa=4*r*(r*r-3)/d**3;phi=2*r/d
        radial=(cr-s[0]**2*ar**2*.5)/(s[0]*kappa)
        gp=(head[1:]-e*(e@head[1:]))/(s[0]*tr)+radial*e
        rec=np.r_[head[0]/s[0]-phi*radial,gp];recB=certificate(s,rec,r,e)
    return dict(physical=s,physical_gradient=g,head_gradient_inf=float(np.linalg.norm(head,np.inf)),radius=r,near_fold=near,normalized_radial_curvature=cr/s[0]**2,trigger=trigger,physical_bound=B,physical_accept=bool(np.isfinite(B) and B<=GAP),recovered_gradient=rec,recovered_bound=recB,recovered_gradient_error=float(np.max(np.abs(rec-g))) if rec is not None else None)

class Oracle:
    def __init__(self,y):
        self.y=y;self.nf=0;self.ng=0;self.nh=0;self.nonfinite=0;self.input_dtypes=set();self.gradient_dtypes=set()
    def fun(self,x):
        self.nf+=1;self.ng+=1;self.input_dtypes.add(str(np.asarray(x).dtype));f,g=fg(x,self.y);self.gradient_dtypes.add(str(g.dtype))
        if not np.isfinite(f) or not np.all(np.isfinite(g)): self.nonfinite+=1
        return f,g
    def hess(self,x):
        self.nh+=1;return hessian(x,self.y)
    def cost(self): return dict(function=self.nf,gradient=self.ng,hessian=self.nh,nonfinite_calls=self.nonfinite,input_dtypes=sorted(self.input_dtypes),gradient_dtypes=sorted(self.gradient_dtypes))

def solve(x,y,method,tol):
    oracle=Oracle(y);options=dict(gtol=tol,maxiter=500)
    if method=='L-BFGS-B': options.update(ftol=0.,maxfun=5000,maxls=40)
    kwargs={'hess':oracle.hess} if method=='trust-exact' else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start=time.perf_counter()
        try:
            res=minimize(oracle.fun,np.asarray(x),jac=True,method=method,options=options,**kwargs)
            row=dict(x=res.x,success=bool(res.success),status=int(res.status),message=str(res.message),nit=int(res.nit),loss=float(res.fun),native_gradient_inf=float(np.linalg.norm(res.jac,np.inf)),result_dtype=str(res.x.dtype),elapsed_seconds=time.perf_counter()-start)
        except Exception as exc:
            row=dict(x=np.asarray(x,dtype=float),success=False,status=-999,message=repr(exc),nit=0,loss=fg(x,y)[0],native_gradient_inf=None,result_dtype=str(np.asarray(x).dtype),elapsed_seconds=time.perf_counter()-start)
    row['counts']=oracle.cost();row['warnings']=dict(collections.Counter(str(w.message) for w in caught))
    return row

def restart(base,y,method,tol):
    x=np.asarray(base['x']);r=np.linalg.norm(x[1:]);direction=x[1:]/r if r else np.array([1.,0.])
    candidates=[x+np.r_[0.,h*direction] for h in [-.01,.01]]+[x]
    losses=[fg(t,y)[0] for t in candidates];j=int(np.argmin(losses));out=solve(candidates[j],y,method,tol)
    out['trial_function_evaluations']=3;out['trial_gradient_evaluations']=3
    out['chosen_offset']=[-.01,.01,0.][j];return out

def observations(row,delta):
    y=np.asarray(row['observations']);angle=row['angle'];degree=row['next_DoLP']
    old=np.r_[1.,degree*np.cos(2*angle),degree*np.sin(2*angle)]
    new=np.r_[1.,degree*np.cos(2*(angle+delta)),degree*np.sin(2*(angle+delta))]
    return y+A@(new-old)

def evaluate_result(res,y,minimum):
    res=dict(res);res['gap']=fg(res['x'],y)[0]-minimum
    res['false_success']=res['success'] and res['gap']>GAP
    res['inaccurate']=not np.isfinite(res['gap']) or res['gap']>GAP
    return res

def summaries(rows):
    groups=collections.defaultdict(list)
    for r in rows: groups[(r['method'],r['gtol'],r['angle_change'])].append(r)
    out=[]
    for (method,tol,delta),rs in groups.items():
        q=dict(method=method,gtol=tol,angle_change=delta,cases=len(rs),previous_failures=sum(not r['previous']['success'] for r in rs))
        for arm in ['baseline','always_restart','triggered_restart']:
            vals=[r[arm] for r in rs]
            q[arm]=dict(false_success=sum(v['false_success'] for v in vals),failures=sum(not v['success'] for v in vals),inaccurate=sum(v['inaccurate'] for v in vals),max_gap=max(v['gap'] for v in vals))
        q['triggers']=sum(r['diagnostic']['trigger'] for r in rs)
        q['false_success_detected']=sum(r['baseline']['false_success'] and r['diagnostic']['trigger'] for r in rs)
        q['accurate_success_triggered']=sum(r['baseline']['success'] and not r['baseline']['inaccurate'] and r['diagnostic']['trigger'] for r in rs)
        q['physical_false_accept']=sum(r['diagnostic']['physical_accept'] and r['baseline']['inaccurate'] for r in rs)
        q['physical_reject_accurate']=sum(not r['diagnostic']['physical_accept'] and not r['baseline']['inaccurate'] for r in rs)
        q['physical_reject_inaccurate']=sum(not r['diagnostic']['physical_accept'] and r['baseline']['inaccurate'] for r in rs)
        q['always_additional_function']=sum(r['always_restart']['counts']['function']+3 for r in rs)
        q['trigger_additional_function']=sum(r['always_restart']['counts']['function']+3 for r in rs if r['diagnostic']['trigger'])
        q['curvature_hessian_queries']=sum(r['diagnostic']['near_fold'] for r in rs)
        out.append(q)
    return out

def run():
    source=json.loads((ROOT/'strategy/warm_start_probe.json').read_text())['records'];rows=[]
    start=time.time();previous_cache={}
    for method in ['BFGS','L-BFGS-B','trust-exact']:
        for tol in [1e-4,1e-6,1e-8]:
            for delta in [0.,.05,.2]:
                for i,row in enumerate(source):
                    yp=np.array(row['previous_observations']);y=observations(row,delta)
                    key=(method,tol,tuple(yp))
                    if key not in previous_cache: previous_cache[key]=solve(np.array([0.,.2,-.1]),yp,method,tol)
                    previous=previous_cache[key];opt,minimum=cone_solution(y)
                    base=evaluate_result(solve(previous['x'],y,method,tol),y,minimum);diag=diagnose(base['x'],y)
                    always=evaluate_result(restart(base,y,method,tol),y,minimum)
                    fixed=always if diag['trigger'] else base
                    rows.append(dict(case=i,seed=row['seed'],angle=row['angle'],noise=row['noise'],next_DoLP=row['next_DoLP'],angle_change=delta,method=method,gtol=tol,observations=y,cone_solution=opt,cone_minimum=minimum,previous=previous,baseline=base,diagnostic=diag,always_restart=always,triggered_restart=fixed))
                print(json.dumps(pack(summaries(rows)[-1])),flush=True)
                save('main_results.json',dict(records=rows,summary=summaries(rows)))
    ref=json.loads((ROOT/'strategy/raw_warm_start_probe.json').read_text())['records']
    subset=[r for r in rows if r['method']=='BFGS' and r['gtol']==1e-6 and r['angle_change']==0]
    reproduction=dict(cases=len(subset),saved_false_success=sum(r['baseline_success'] and r['baseline_gap']>GAP for r in ref),new_false_success=sum(r['baseline']['false_success'] for r in subset),saved_failures=sum(not r['baseline_success'] for r in ref),new_failures=sum(not r['baseline']['success'] for r in subset),saved_ids=[i for i,r in enumerate(ref) if r['baseline_success'] and r['baseline_gap']>GAP],new_ids=[r['case'] for r in subset if r['baseline']['false_success']],max_previous_coordinate_difference=max(np.max(np.abs(np.array(ref[r['case']]['previous_coordinates'])-r['previous']['x'])) for r in subset),max_loss_gap_difference=max(abs(ref[r['case']]['baseline_gap']-r['baseline']['gap']) for r in subset),new_after_restart_false_success=sum(r['triggered_restart']['false_success'] for r in subset),new_after_restart_failures=sum(not r['triggered_restart']['success'] for r in subset))
    save('reproduction.json',reproduction)
    promotion=solve(np.array([0.,.2,-.1],dtype=np.float32),np.array(source[0]['previous_observations']),'BFGS',1e-6)
    save('scipy_dtype_probe.json',promotion)
    save('environment.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),platform=platform.platform(),python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,elapsed_seconds=time.time()-start,previous_unique_runs=len(previous_cache),main_next_runs=len(rows),protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.json').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('REPRODUCTION',json.dumps(pack(reproduction)),flush=True)

if __name__=='__main__': run()
