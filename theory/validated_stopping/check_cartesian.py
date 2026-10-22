"""Exact Cartesian derivative checks, independent of radial formulas.

Set a=0 (I=1 exactly). Fractions then compute the entire physical/head
Jacobian and Hessian without transcendental values or radial square roots.
This is a formula unit check, not an additional optimizer experiment.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,sys
from monitor import evaluate
from run import stress_cases

P=Path(__file__).resolve().parent;R=P.parents[1]


def derivatives(x,y):
    w=list(map(F,x[1:]));y=list(map(F,y));q=sum(t*t for t in w)
    if q>1:w=[t/q for t in w];q=1/q
    d=1+q;pol=[2*t/d for t in w]
    original=list(map(F,x[1:]));od=1+sum(t*t for t in original)
    assert pol==[2*t/od for t in original]
    z=[sum(y)/2,y[0]-y[2],y[1]-y[3]]
    gp=[(pol[i]-z[i+1])/2 for i in range(2)];gI=1-z[0]
    J=[[2*int(i==j)/d-4*w[i]*w[j]/d**2 for j in range(2)] for i in range(2)]
    b=gI+sum(pol[i]*gp[i] for i in range(2))
    j=[sum(J[i][k]*gp[i] for i in range(2)) for k in range(2)]
    eps2=max(b*b,sum(t*t for t in j))
    if q<F(1,4):return eps2,None
    H=[]
    for j in range(2):
        row=[]
        for k in range(2):
            chain=sum((F(-4)*(int(i==j)*w[k]+int(i==k)*w[j]+int(j==k)*w[i])/d**2+
                       16*w[i]*w[j]*w[k]/d**3)*gp[i] for i in range(2))
            row.append(sum(J[i][j]*J[i][k] for i in range(2))/2+chain)
        H.append(row)
    cr=sum(w[j]*H[j][k]*w[k] for j in range(2) for k in range(2))/q
    return eps2,max(F(0),-cr)


def main():
    inputs=[r for r in stress_cases() if r['family']!='unsupported_input' and r['x'][0]==0.]
    streams=json.loads((R/'application/real_data/INPUTS.json').read_text());ys={x['stream']:x['observations'] for x in streams}
    rows=[x for x in json.loads((R/'application/real_data/RESULTS.json').read_text())['records'] if x['arm']=='warm_start']
    for k in range(32):
        r=rows[(len(rows)-1)*k//31];inputs.append(dict(x=[0.,*r['final_x'][1:]],y=ys[r['stream']][r['index']]))
    checked=inconclusive=0
    for row in inputs:
        eps2,nu=derivatives(row['x'],row['y'])
        for bits in [32,64,128,256]:
            result=evaluate(row['x'],row['y'],bits)
            if result['status']=='inconclusive':inconclusive+=1;continue
            assert F(result['epsilon_upper'])**2>=eps2,(row,bits,'head gradient')
            if nu is not None:assert F(result['nu_upper'])>=nu,(row,bits,'curvature')
            checked+=1
    output=dict(inputs=len(inputs),checked=checked,inconclusive=inconclusive,
        method='Exact Fraction Cartesian Jacobian/Hessian at I=1; compared with outward radial tolerance enclosures.',
        source_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['arithmetic.py','monitor.py',Path(__file__).name]})
    path=Path(sys.argv[1]) if len(sys.argv)>1 else P/'results/CARTESIAN_CHECKS.json'
    path.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))


if __name__=='__main__':main()
