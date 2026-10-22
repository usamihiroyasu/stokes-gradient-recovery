"""Independent PyTorch sheet-symmetry checks and NumPy physical-step audit.

Uses frozen historical endpoints. Does not import the candidate algorithm/core.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
P=Path(__file__).resolve().parent;R=P.parents[1]
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
history=json.loads((R/'application/real_data/RESULTS.json').read_text())['records']
warm=[r for r in history if r['arm']=='warm_start']
bad=[r for r in warm if r['false_success']]
controls=[warm[i] for i in np.linspace(0,len(warm)-1,32,dtype=int)]
selected=bad+controls
selection=[dict(stream=r['stream'],index=r['index'],method=r['method'],original_false_success=r['false_success']) for r in selected]
target=P/'INDEPENDENT_SELECTION.json'
if target.exists():assert json.loads(target.read_text())==selection
else:target.write_text(json.dumps(selection,indent=2)+'\n')
streams=json.loads((R/'application/real_data/INPUTS.json').read_text())
ys={s['stream']:np.asarray(s['observations']) for s in streams}
A=np.array([[.5,.5,0],[.5,0,.5],[.5,-.5,0],[.5,0,-.5]])
W=np.diag([1.,.5,.5]);M=np.diag([1.,2.,2.])

def mapped(x):
    I=torch.exp(x[0]);w=x[1:]
    return torch.cat((I.reshape(1),I*2*w/(1+w.square().sum())))

def projection(v):
    t=v[0];p=v[1:];rho=np.linalg.norm(p)
    if t>=rho:return v.copy()
    if t<=-rho:return np.zeros(3)
    return .5*(t+rho)*np.r_[1.,p/rho]

def bound(s,z):
    g=W@(s-z);rho=np.linalg.norm(s[1:])
    if rho==0:return .5*g@M@g
    n=np.r_[-1.,s[1:]/rho];slack=max(0.,-n@s)
    lam=max(0.,-(n@M@g+slack)/(n@M@n))
    return .5*(g+lam*n)@M@(g+lam*n)+lam*slack

max_state=max_gradient=max_increase=max_contraction_excess=max_bound_violation=0.
max_lift_error=0.;max_pg_updates=0;pg_inaccurate=0
for row in selected:
    x=torch.tensor(row['final_x'],requires_grad=True);y=ys[row['stream']][row['index']]
    r=float(torch.linalg.vector_norm(x[1:]).detach());r2=r*r
    inner=x.detach().clone()
    if r>1:inner[1:]/=r2
    inner.requires_grad_(True)
    f=lambda xx:.5*((torch.tensor(A)@mapped(xx)-torch.tensor(y))**2).sum()
    s=mapped(x).detach().numpy();si=mapped(inner).detach().numpy()
    gx=torch.autograd.grad(f(x),x)[0].detach().numpy()
    gi=torch.autograd.grad(f(inner),inner)[0].detach().numpy()
    max_state=max(max_state,float(np.max(np.abs(s-si))))
    if r>1:
        e=np.asarray(row['final_x'][1:])/r;H=np.eye(2)-2*np.outer(e,e)
        max_gradient=max(max_gradient,float(np.max(np.abs(gi[1:]-r2*H@gx[1:]))))
    z=np.array([sum(y)/2,y[0]-y[2],y[1]-y[3]]);rho=np.linalg.norm(z[1:])
    optimum=z if z[0]>=rho else np.zeros(3) if 2*z[0]+rho<=0 else ((2*z[0]+rho)/3)*np.r_[1.,z[1:]/rho]
    for k in range(65):
        b=bound(s,z);gap=.5*(s-z)@W@(s-z)-.5*(optimum-z)@W@(optimum-z)
        max_bound_violation=max(max_bound_violation,float(gap-b))
        if b<=1e-6:break
        if k==64:break
        new=projection(s-W@(s-z))
        increase=.5*(new-z)@W@(new-z)-.5*(s-z)@W@(s-z)
        max_increase=max(max_increase,float(increase))
        max_contraction_excess=max(max_contraction_excess,float(np.linalg.norm(new-optimum)-.5*np.linalg.norm(s-optimum)))
        assert new[0]>=np.linalg.norm(new[1:])-1e-14
        s=new
    max_pg_updates=max(max_pg_updates,k);pg_inaccurate+=int(gap>1e-6)
    if s[0]>0:
        q=s[1:]/s[0];d2=min(1.,max(0.,float(q@q)));xx=np.r_[np.log(s[0]),q/(1+np.sqrt(1-d2))]
        max_lift_error=max(max_lift_error,float(np.max(np.abs(mapped(torch.tensor(xx)).detach().numpy()-s))))
assert max_state<1e-14 and max_gradient<1e-13
assert max_increase<1e-14 and max_contraction_excess<1e-13 and max_bound_violation<1e-13
assert pg_inaccurate==0 and max_lift_error<1e-13
out=dict(endpoints=len(selected),original_false_successes=len(bad),controls=len(controls),
    selection_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),torch=torch.__version__,
    max_output_difference=max_state,max_gradient_transform_difference=max_gradient,
    max_projected_step_objective_increase=max_increase,max_half_contraction_excess=max_contraction_excess,
    max_gap_above_bound=max_bound_violation,max_pg_updates=max_pg_updates,
    inaccurate_after_physical_steps=pg_inaccurate,max_physical_lift_difference=max_lift_error,
    independent_paper_review=False)
(P/'INDEPENDENT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
