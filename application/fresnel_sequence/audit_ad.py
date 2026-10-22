"""Independently audit saved pilot endpoints with PyTorch float64 AD."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
torch.set_num_threads(1)
data=json.loads((HERE/'results.json').read_text());inputs=json.loads((HERE/'inputs.json').read_text())
rows=[r for r in data['records'] if r['arm']=='baseline' and r['final']['false_success']]
# Complete false-success set plus the first endpoint in every sequence for both solvers.
rows += [r for r in data['records'] if r['arm']=='baseline' and r['frame']==0]
A=torch.tensor([[.5,.5,0],[.5,0,.5],[.5,-.5,0],[.5,0,-.5]],dtype=torch.float64)
audit=[]
for r in rows:
    x=torch.tensor(r['initial']['x'],dtype=torch.float64,requires_grad=True)
    y=torch.tensor(inputs[r['sequence']]['observations'][r['frame']],dtype=torch.float64)
    def fun(z):
        I=torch.exp(z[0]);w=z[1:];s=torch.cat([I.reshape(1),I*2*w/(1+w@w)])
        residual=A@s-y
        return .5*(residual@residual)
    loss=fun(x);g=torch.autograd.grad(loss,x)[0];H=torch.autograd.functional.hessian(fun,x)
    direction=x[1:].detach()/torch.linalg.norm(x[1:].detach());v=torch.cat([torch.zeros(1,dtype=torch.float64),direction]);cr=float((v@H@v).detach()/torch.exp(2*x[0].detach()))
    ge=abs(float(torch.max(torch.abs(g)).detach())-r['initial']['native_gradient_inf'])
    he=abs(cr-r['initial_diagnostic']['normalized_radial_curvature'])
    le=abs(float(loss.detach())-r['initial']['loss'])
    assert ge<1e-12 and he<1e-12 and le<1e-12
    audit.append(dict(method=r['method'],sequence=r['sequence'],frame=r['frame'],gradient_inf_difference=ge,normalized_radial_curvature_difference=he,loss_difference=le))
out={'torch_version':torch.__version__,'dtype':'float64','endpoints':len(audit),'complete_false_success_set_audited':len([r for r in rows if r['final']['false_success']]),'max_gradient_inf_difference':max(r['gradient_inf_difference'] for r in audit),'max_curvature_difference':max(r['normalized_radial_curvature_difference'] for r in audit),'max_loss_difference':max(r['loss_difference'] for r in audit),'results_sha256':hashlib.sha256((HERE/'results.json').read_bytes()).hexdigest(),'records':audit}
(HERE/'independent_AD_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
