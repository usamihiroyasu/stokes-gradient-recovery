"""Independent automatic derivatives for all 600 fixed theory objectives."""
from pathlib import Path
import json
import numpy as np
import torch
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
d=np.load(P/'results/AD_REFERENCE.npz');maximum=dict(b=0.,j=0.,cr=0.)
for i,x0 in enumerate(d['x']):
    x=torch.tensor(x0,requires_grad=True);z=torch.tensor(d['z'][i]);W=torch.tensor(d['W'][i]);beta=float(d['beta'][i])
    def f(xx):
        I=torch.exp(xx[0]);w=xx[1:];s=torch.cat((I.reshape(1),I*2*w/(1+w.square().sum())))
        v=s-z;t=torch.tensor([1.,.3,-.2])@v
        return .5*v@W@v+beta*t**4/4
    g=torch.autograd.functional.jacobian(f,x);H=torch.autograd.functional.hessian(f,x)
    radius=np.linalg.norm(x0[1:])
    # At w=0 the coordinates erase the prescribed direction. Recover that
    # metadata from the frozen Cartesian input grid (angle before mode).
    angle=[0.,.7][(i//4)%2]
    e=x0[1:]/radius if radius else np.array([np.cos(angle),np.sin(angle)])
    v=torch.tensor(np.r_[0.,e])
    maximum['b']=max(maximum['b'],abs(float((g[0]/torch.exp(x[0])).detach())-d['b'][i]))
    maximum['j']=max(maximum['j'],float(np.max(np.abs(g[1:].detach().numpy()-d['j'][i]))))
    maximum['cr']=max(maximum['cr'],abs(float(v@H@v)-d['cr'][i]))
assert max(maximum.values())<1e-10
out=dict(objectives=len(d['x']),torch=torch.__version__,max_absolute_derivative_discrepancy=maximum,
         origin_directions_restored_from_frozen_input_grid=int(sum(np.linalg.norm(x[1:])==0 for x in d['x'])),independent_manuscript_review=False)
(P/'results/INDEPENDENT_AD_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
