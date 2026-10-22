"""CPU eager-AD acquisition and recovery on one shared quartic objective.

All curvature acquisition is timed. Direct physical differentiation is included.
Scalar curvature is contracted from a reverse-over-reverse HVP in this code.
"""
from pathlib import Path
from decimal import Decimal as D,localcontext
import argparse,hashlib,json,math,platform,statistics,subprocess,time
import numpy as np
import torch
from corrected_recovery_checks import loss_grad

HERE=Path(__file__).resolve().parent
METHODS=['direct','first','vector','mixed']

def objective(s):
    x,y,z=s; q=y+.4*z
    return .2*x**4+.1*q**4+.3*x*y*z+.05*z**3+.4*x+.2*y-.3*z

def acquire(method,dtype,I0,r0):
    a=torch.tensor(math.log(I0),dtype=dtype,requires_grad=True)
    w=torch.tensor([.6*r0,.8*r0],dtype=dtype,requires_grad=True)
    I=a.exp();F=2*w/(1+(w*w).sum());s=torch.cat((I.reshape(1),I*F));f=objective(s)
    if method=='direct':
        return torch.autograd.grad(f,s)[0].detach(),s.detach(),None
    r=torch.linalg.vector_norm(w.detach());e=w.detach()/r;P=torch.eye(2,dtype=dtype)-torch.outer(e,e)
    phi=2*r/(1+r*r);ar=2*(1-r*r)/(1+r*r)**2;t=2/(1+r*r)
    if method=='first':
        if ar.item()==0:return None,s.detach(),float(r)
        da,j=torch.autograd.grad(f,(a,w));b=da/I
        gp=e*torch.dot(e,j)/(I*ar)+(P@j)/(I*t)
    else:
        if ar.item()==0:
            da,j=torch.autograd.grad(f,(a,w),create_graph=True)
            v=torch.autograd.grad(torch.dot(j,e),w)[0]
            hp=torch.zeros(2,dtype=dtype)
        else:
            da,j,g=torch.autograd.grad(f,(a,w,s),create_graph=True)
            v=torch.autograd.grad(torch.dot(j,e),w,retain_graph=True)[0]
            # The required physical action is acquired and timed off the fold.
            hp=torch.autograd.grad(torch.dot(g[1:],e),s)[0][1:]
        b=da/I
        k=4*r*(r*r-3)/(1+r*r)**3
        if method=='vector':
            J=ar*torch.outer(e,e)+t*P
            ell=-4*r/(1+r*r)**2;K=k*torch.outer(e,e)+ell*P
            gp=torch.linalg.solve(K,v/I-I*ar*(J@hp))
        elif method=='mixed':
            cr=torch.dot(e,v);hrr=torch.dot(e,hp)
            m=(cr/I-I*ar*ar*hrr)/k
            gp=(P@j)/(I*t)+e*m
        else:raise ValueError(method)
    result=torch.cat(((b-phi*torch.dot(e,gp)).reshape(1),gp))
    return result.detach(),s.detach(),float(r)

def reference(s):
    with localcontext() as ctx:
        ctx.prec=80
        values=[D.from_float(float(x)) for x in s.tolist()]
        return np.array([float(x) for x in loss_grad(values,'quartic')[1]])

def accuracy_records():
    rows=[]
    radii=[1.]+[1+sign*10.**(-k) for k in [1,3,5,7] for sign in [-1,1]]
    for dtype_name,dtype in [('float32',torch.float32),('float64',torch.float64)]:
        for intensity in [.02,1.2]:
            for r in radii:
                for method in METHODS:
                    estimate,s,actual_r=acquire(method,dtype,intensity,r)
                    truth=reference(s)
                    row=dict(dtype=dtype_name,I_requested=intensity,r_requested=r,method=method,
                             physical_state=s.tolist(),actual_radius=actual_r)
                    if estimate is None:
                        row.update(status='singular_first_inverse',gradient=None,error=None)
                    else:
                        g=estimate.double().numpy();assert np.all(np.isfinite(g))
                        error=float(np.linalg.norm(g-truth))
                        row.update(status='evaluated',gradient=g.tolist(),error=error)
                        if dtype_name=='float64' and method!='first':assert error<1e-10
                    rows.append(row)
    return rows

def timings():
    rows=[];repeats=20;batches=7;warmup=5
    for dtype_name,dtype in [('float32',torch.float32),('float64',torch.float64)]:
        for r in [.999,1.,1.001]:
            methods=[m for m in METHODS if not (m=='first' and r==1.)]
            values={m:[] for m in methods}
            for m in methods:
                for _ in range(warmup):acquire(m,dtype,1.2,r)
            for batch in range(batches):
                order=methods[batch%len(methods):]+methods[:batch%len(methods)]
                for m in order:
                    start=time.perf_counter_ns()
                    for _ in range(repeats):acquire(m,dtype,1.2,r)
                    values[m].append((time.perf_counter_ns()-start)/repeats/1000.)
            for m in methods:
                values_m=values[m]
                rows.append(dict(dtype=dtype_name,r=r,I=1.2,method=m,batch_mean_us=values_m,
                    median_us=statistics.median(values_m),q25_us=float(np.quantile(values_m,.25)),
                    q75_us=float(np.quantile(values_m,.75))))
    return rows,dict(repeats_per_batch=repeats,batches=batches,warmup_calls=warmup,rotating_method_order=True)

def summarize(accuracy,timing):
    errors={dtype:{m:max(x['error'] for x in accuracy if x['dtype']==dtype and x['method']==m and x['error'] is not None)
                  for m in METHODS} for dtype in ['float32','float64']}
    medians={x['method']:x['median_us'] for x in timing if x['dtype']=='float64' and x['r']==.999}
    return dict(accuracy_rows=len(accuracy),evaluated=sum(x['status']=='evaluated' for x in accuracy),
                singular_rows=sum(x['status']!='evaluated' for x in accuracy),timing_rows=len(timing),
                max_error_by_dtype=errors,median_us_float64_r0999=medians)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    accuracy=accuracy_records(); path=HERE/'results/ad_route_benchmark.json'
    if args.check_only:
        prior=json.loads(path.read_text());assert prior['accuracy_records']==accuracy
        assert prior['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        print(json.dumps(dict(status='PASS',accuracy_rows=len(accuracy),timings_repeated=False)));return
    timing,protocol=timings()
    hardware=subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True).stdout.strip() if platform.system()=='Darwin' else platform.processor()
    report=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='One CPU eager-AD graph and quartic loss; curvature acquisition included; no optimizer or learned reconstruction.',
        scalar_implementation='c=e dot HVP and hrr=e dot physical Hessian action; physical curvature query omitted at the exact fold; no scalar-kernel cost advantage assumed.',
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                         machine=platform.machine(),hardware=hardware,system=platform.platform(),threads=1,device='cpu'),
        protocol=protocol,accuracy_records=accuracy,timing_records=timing,summary=summarize(accuracy,timing))
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
