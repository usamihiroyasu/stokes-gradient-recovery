"""Directional sensitivity of supplied curvature action at fixed geometry.

Decimal matrix-based recovery is compared with closed directional coefficients.
These are consistency checks of derivative observations, not application data.
"""
from pathlib import Path
from decimal import Decimal as D, localcontext
import hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent

def dot(x, y):
    return sum(a*b for a,b in zip(x,y))

def mv(A, x):
    return [dot(row,x) for row in A]

def inverse(A):
    det=A[0][0]*A[1][1]-A[0][1]*A[1][0]
    return [[A[1][1]/det,-A[0][1]/det],[-A[1][0]/det,A[0][0]/det]]

def matrices(r,e):
    w=[r*x for x in e]; den=1+dot(w,w); we=dot(w,e)
    J=[[(2*D(i==j)/den-4*w[i]*w[j]/den**2) for j in range(2)] for i in range(2)]
    K=[[(-4*we*D(i==j)/den**2-4*(e[i]*w[j]+w[i]*e[j])/den**2
         +16*we*w[i]*w[j]/den**3) for j in range(2)] for i in range(2)]
    return J,K

def main():
    records=[]
    with localcontext() as ctx:
        ctx.prec=80
        I=D('1.2'); eps=D('0.1'); e=[D('0.6'),D('0.8')]; ep=[-e[1],e[0]]
        b=D('0.31'); v=[D('0.7'),D('-0.2')]; h=[D('0.4'),D('0.6')]
        grid=[('fold',D(1),0)]+[(side,1+sign*D(10)**(-k),k)
             for k in range(1,9) for side,sign in [('inner',-1),('outer',1)]]
        for side,r,k in grid:
            J,K=matrices(r,e); ki=inverse(K); a=dot(e,mv(J,e)); phi=2*r/(1+r*r)
            def recover(hh):
                Jh=mv(J,hh); gp=mv(ki,[v[i]/I-I*a*Jh[i] for i in range(2)])
                return [b-phi*dot(e,gp)]+gp
            base=recover(h)
            beta_rad=I*(1-r*r)**2/(r*(3-r*r)*(1+r*r))
            beta_tan=I*(1-r*r)/(r*(1+r*r))
            for direction,d,beta in [('radial',e,beta_rad),('tangential',ep,beta_tan)]:
                shifted=recover([h[i]+eps*d[i] for i in range(2)])
                change=[shifted[i]-base[i] for i in range(3)]
                predicted=[-phi*beta*eps*dot(e,d)]+[beta*eps*x for x in d]
                residual=max(abs(x-y) for x,y in zip(change,predicted))
                assert residual<D('1e-65')
                if r==1: assert change==[D(0)]*3
                measured_gp=dot(change[1:],change[1:]).sqrt()/(I*eps)
                measured_full=dot(change,change).sqrt()/eps
                expected_full=abs(beta)*((1+phi*phi).sqrt() if direction=='radial' else 1)
                full_residual=abs(measured_full-expected_full)
                assert full_residual<D('1e-65')
                records.append(dict(side=side,r=float(r),distance=float(abs(r-1)),k=k,
                    direction=direction,beta=float(beta),beta_decimal=str(beta),
                    normalized_gain=float(abs(beta)/I),measured_gain=float(measured_gp),
                    full_gradient_gain=float(measured_full),identity_residual=float(residual),
                    identity_residual_decimal=str(residual),full_gain_residual=float(full_residual)))
        assert len(records)==34
    slopes={}
    for side in ['inner','outer']:
        for direction,expected in [('radial',2),('tangential',1)]:
            rr=[x for x in records if x['side']==side and x['direction']==direction and 4<=x['k']<=8]
            assert len(rr)==5 and all(x['measured_gain']>0 for x in rr)
            slope=float(np.polyfit(np.log10([x['distance'] for x in rr]),
                                   np.log10([x['measured_gain'] for x in rr]),1)[0])
            assert abs(slope-expected)<0.001
            slopes[side+'_'+direction]=slope
    report=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Fixed geometry; 80-digit Decimal before/after recovery. Curvature-action input varies; b and v fixed.',
        precision=80,I=1.2,epsilon_h=0.1,e=[0.6,0.8],b=0.31,v=[0.7,-0.2],h=[0.4,0.6],
        records=records,slopes=slopes,
        summary=dict(cases=len(records),max_identity_residual=max(x['identity_residual'] for x in records),
                     max_full_gain_residual=max(x['full_gain_residual'] for x in records),slopes=slopes))
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/anisotropic_curvature_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':
    main()
