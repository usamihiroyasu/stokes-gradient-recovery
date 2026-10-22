"""Independent PyTorch AD and real float32/64 LBFGS; no SciPy imports."""
from pathlib import Path
import collections, datetime, hashlib, json, platform, sys, time, warnings
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
torch.set_num_threads(1)
A64=np.array([[.5,.5,0],[.5,0,.5],[.5,-.5,0],[.5,0,-.5]])

def native_loss(x,y):
    a=x.new_tensor([[.5,.5,0],[.5,0,.5],[.5,-.5,0],[.5,0,-.5]])
    intensity=x[0].exp();w=x[1:];polarization=2*w/(1+w.square().sum())
    s=torch.cat((intensity[None],intensity*polarization));residual=a@s-y
    return residual.square().sum()/2

def save(name,value): (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def solve(x0,observations,dtype):
    x=torch.tensor(x0,dtype=dtype,requires_grad=True);y=torch.tensor(observations,dtype=dtype)
    optim=torch.optim.LBFGS([x],lr=1.,max_iter=500,max_eval=1000,tolerance_grad=1e-6,tolerance_change=0.,history_size=20,line_search_fn='strong_wolfe')
    count=0;nonfinite=0;oracle_dtypes=set();gradient_dtypes=set();start=time.perf_counter();error=None
    def closure():
        nonlocal count,nonfinite
        count+=1;optim.zero_grad();loss=native_loss(x,y);loss.backward();oracle_dtypes.add(str(loss.dtype));gradient_dtypes.add(str(x.grad.dtype))
        if not bool(torch.isfinite(loss) and torch.isfinite(x.grad).all()): nonfinite+=1
        return loss
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try: optim.step(closure)
        except Exception as exc: error=repr(exc)
    loss=native_loss(x,y);grad=torch.autograd.grad(loss,x)[0];norm=float(grad.abs().max())
    state=optim.state[x]
    return dict(x=x.detach().double().tolist(),success=error is None and np.isfinite(norm) and norm<=1e-6,status='gradient_accepted' if error is None and norm<=1e-6 else 'not_gradient_accepted',message=error,native_gradient_inf=norm if np.isfinite(norm) else None,loss=float(loss.detach()) if bool(torch.isfinite(loss)) else None,nit=state.get('n_iter'),function=count,gradient=count,nonfinite_calls=nonfinite,input_dtype=str(x.dtype),output_dtype=str(x.dtype),oracle_dtypes=sorted(oracle_dtypes),gradient_dtypes=sorted(gradient_dtypes),elapsed_seconds=time.perf_counter()-start,warnings=dict(collections.Counter(str(w.message) for w in caught)))

def fg64(x,y):
    # Direct AD at represented coordinates, with original float64 observations.
    x=torch.tensor(x,dtype=torch.float64,requires_grad=True);y=torch.tensor(y,dtype=torch.float64)
    loss=native_loss(x,y);g=torch.autograd.grad(loss,x)[0]
    return float(loss.detach()),g.detach().numpy()

def optimum(y):
    z=np.array([sum(y)/2,y[0]-y[2],y[1]-y[3]]);rho=np.linalg.norm(z[1:])
    if z[0]>=rho:s=z
    elif 2*z[0]+rho<=0:s=np.zeros(3)
    else:
        intensity=(2*z[0]+rho)/3;s=np.r_[intensity,intensity*z[1:]/rho]
    return .5*float(np.sum((A64@s-y)**2))

def run_precision():
    source=json.loads((ROOT/'strategy/warm_start_probe.json').read_text())['records'];rows=[];cache={}
    for dtype in [torch.float32,torch.float64]:
        for delta in [0.,.05,.2]:
            for i,row in enumerate(source):
                yp=row['previous_observations'];key=(str(dtype),tuple(yp))
                if key not in cache:cache[key]=solve([0.,.2,-.1],yp,dtype)
                previous=cache[key];angle=row['angle'];degree=row['next_DoLP']
                old=np.r_[1.,degree*np.cos(2*angle),degree*np.sin(2*angle)];new=np.r_[1.,degree*np.cos(2*(angle+delta)),degree*np.sin(2*(angle+delta))]
                y=np.array(row['observations'])+A64@(new-old);minimum=optimum(y)
                base=solve(previous['x'],y,dtype);gap=fg64(base['x'],y)[0]-minimum
                # Radial curvature audit in float64 at the represented native-precision endpoint.
                t=torch.tensor(base['x'],dtype=torch.float64,requires_grad=True);ty=torch.tensor(y,dtype=torch.float64)
                h=torch.autograd.functional.hessian(lambda u:native_loss(u,ty),t).detach().numpy()
                xx=np.array(base['x']);r=np.linalg.norm(xx[1:]);e=xx[1:]/r if r else np.array([1.,0.]);v=np.r_[0.,e]
                c=float(v@h@v/np.exp(2*xx[0]));trigger=bool(abs(r-1)<=1e-3 and c < -1e-7)
                candidates=[xx+np.r_[0.,d*e] for d in [-.01,.01]]+[xx]
                candidate=min(candidates,key=lambda u:fg64(u,y)[0]);always=solve(candidate,y,dtype)
                for result in [base,always]:
                    result['gap64']=fg64(result['x'],y)[0]-minimum;result['false_success']=bool(result['success'] and result['gap64']>1e-6);result['inaccurate']=bool(result['gap64']>1e-6)
                rows.append(dict(case=i,dtype=str(dtype),angle_change=delta,seed=row['seed'],angle=angle,noise=row['noise'],next_DoLP=degree,observations=y.tolist(),previous=previous,baseline=base,radius=float(r),normalized_radial_curvature=c,trigger=trigger,always_restart=always,triggered_restart=always if trigger else base,trial_float64_function_gradient_evaluations=3))
            summary=[]
            for dt in ['torch.float32','torch.float64']:
                for dd in [0.,.05,.2]:
                    rs=[r for r in rows if r['dtype']==dt and r['angle_change']==dd]
                    if not rs:continue
                    q=dict(dtype=dt,angle_change=dd,cases=len(rs),previous_failures=sum(not r['previous']['success'] for r in rs),triggers=sum(r['trigger'] for r in rs),detected_false_success=sum(r['trigger'] and r['baseline']['false_success'] for r in rs),accurate_success_triggered=sum(r['trigger'] and r['baseline']['success'] and not r['baseline']['inaccurate'] for r in rs))
                    for arm in ['baseline','always_restart','triggered_restart']:
                        vals=[r[arm] for r in rs];q[arm]=dict(false_success=sum(v['false_success'] for v in vals),failures=sum(not v['success'] for v in vals),inaccurate=sum(v['inaccurate'] for v in vals),max_gap=max(v['gap64'] for v in vals))
                    summary.append(q)
            save('precision_results.json',dict(records=rows,summary=summary,metadata=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),threads=torch.get_num_threads(),previous_unique_runs=len(cache),operational_success='Native gradient_inf<=1e-6; no native solver success flag',restart_trial_and_curvature_precision='float64 diagnostic controller',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
            print(json.dumps(summary[-1]),flush=True)

def run_ad():
    # Exchange AD values at frozen saved endpoints for the SciPy-side independent audit.
    ref=json.loads((ROOT/'strategy/raw_warm_start_probe.json').read_text())['records'];records=[]
    for i,row in enumerate(ref):
        x=torch.tensor(row['baseline_coordinates'],dtype=torch.float64,requires_grad=True);y=torch.tensor(row['observations'],dtype=torch.float64)
        loss=native_loss(x,y);gradient=torch.autograd.grad(loss,x)[0];h=torch.autograd.functional.hessian(lambda t:native_loss(t,y),x)
        records.append(dict(case=i,x=x.detach().tolist(),y=y.tolist(),loss=float(loss.detach()),gradient=gradient.tolist(),hessian=h.tolist()))
    save('independent_torch_ad.json',dict(records=records,torch=torch.__version__,python=sys.version,numpy=np.__version__))

if __name__=='__main__':
    run_ad()
    if '--ad-only' not in sys.argv:run_precision()
