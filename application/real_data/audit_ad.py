"""Independent float64 PyTorch checks, including every cold/warm false success."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
P=Path(__file__).resolve().parent;torch.set_num_threads(1)
d=np.load(P/'ad_reference.npz',allow_pickle=False)
A=torch.tensor([[.5,.5,0],[.5,0,.5],[.5,-.5,0],[.5,0,-.5]],dtype=torch.float64)
errors=[]
for i,(xx,yy) in enumerate(zip(d['x'],d['y'])):
    x=torch.tensor(xx,dtype=torch.float64,requires_grad=True);y=torch.tensor(yy,dtype=torch.float64)
    def f(z):
        I=torch.exp(z[0]);w=z[1:];s=torch.cat([I.reshape(1),I*2*w/(1+w@w)]);v=A@s-y;return .5*(v@v)
    g=torch.autograd.grad(f(x),x)[0].detach().numpy();H=torch.autograd.functional.hessian(f,x).detach().numpy()
    s=torch.cat([torch.exp(x[0]).reshape(1),torch.exp(x[0])*2*x[1:]/(1+x[1:]@x[1:])]).detach().requires_grad_(True)
    loss=.5*torch.sum((A@s-y)**2);physical=torch.autograd.grad(loss,s)[0].detach().numpy()
    errors.append({'index':i,'head_gradient_max_difference':float(np.max(np.abs(g-d['g'][i]))),'Hessian_max_difference':float(np.max(np.abs(H-d['H'][i]))),'physical_gradient_max_difference':float(np.max(np.abs(physical-d['physical'][i])))})
for e in errors:assert max(e[k] for k in e if k!='index')<1e-11
out={'torch':torch.__version__,'dtype':'float64','endpoints':len(errors),'false_successes':sum(r['false_success'] for r in json.loads((P/'AD_SELECTION.json').read_text())),'all_cold_and_warm_false_successes_audited':True,'max_head_gradient_difference':max(r['head_gradient_max_difference'] for r in errors),'max_Hessian_difference':max(r['Hessian_max_difference'] for r in errors),'max_physical_gradient_difference':max(r['physical_gradient_max_difference'] for r in errors),'source_results_sha256':hashlib.sha256((P/'RESULTS.json').read_bytes()).hexdigest(),'records':errors}
(P/'INDEPENDENT_AD_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
