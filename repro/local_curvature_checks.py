"""Local C2 curvature recovery and supplied curvature-input perturbations.

One retained nonquadratic loss; no optimizer, training, or new image data.
Reporting metric is identity, independently of the physical Hessian.
"""
from pathlib import Path
from decimal import Decimal as D, localcontext
import hashlib, json
import numpy as np
from corrected_recovery_checks import dvec, dot, loss_grad, head_values

HERE=Path(__file__).resolve().parent

def main():
    I=1.2;e=np.array([np.cos(.7),np.sin(.7)])
    projector=np.outer(e,e);P=np.eye(2)-projector
    records=[];bias=[]
    with localcontext() as ctx:
        ctx.prec=80
        ed=dvec(e);ed=[x/dot(ed,ed).sqrt() for x in ed]
        for r in [.9,.99,.9999,1.,1.0001,1.01,1.1]:
            rd=D(str(r));Id=D(str(I));w=[rd*x for x in ed]
            physical=lambda s:loss_grad(s,'quartic')
            f,bd,jd,gd=head_values(Id,w,physical)
            den=1+dot(w,w);s=[Id]+[Id*2*x/den for x in w]
            x,y,z=s;q=y+D('.4')*z
            Hd=[[D('1.2')*q*q,D('.48')*q*q+D('.3')*x],
                [D('.48')*q*q+D('.3')*x,D('.192')*q*q+D('.3')*z]]
            # Differentiate the physical gradient to check Hpp independently.
            step=D('1e-18');hessian_error=D(0)
            for col in range(2):
                sm=s.copy();sp=s.copy();sm[col+1]-=step;sp[col+1]+=step
                gm=physical(sm)[1];gp=physical(sp)[1]
                for row in range(2):
                    hessian_error=max(hessian_error,abs((gp[row+1]-gm[row+1])/(2*step)-Hd[row][col]))
            hm=head_values(Id,[(rd-step)*x for x in ed],physical)[2]
            hp=head_values(Id,[(rd+step)*x for x in ed],physical)[2]
            v=np.array([float((hp[k]-hm[k])/(2*step)) for k in range(2)])
            H=np.array([[float(t) for t in row] for row in Hd]);g=np.array([float(x) for x in gd]);b=float(bd)
            phi=2*r/(1+r*r);a=2*(1-r*r)/(1+r*r)**2;t=2/(1+r*r)
            J=a*projector+t*P
            K=4*r*(r*r-3)/(1+r*r)**3*projector-4*r/(1+r*r)**2*P
            ki=np.linalg.inv(K);A=np.vstack([-phi*e,np.eye(2)])
            R=A@ki/I;C=-I*a*A@ki@J
            def recover(bb,vv,hh):
                pp=ki@(vv/I-I*a*J@hh)
                return np.r_[bb-phi*e@pp,pp]
            base=recover(b,v,H@e)
            error=float(np.linalg.norm(base-g))
            assert error<2e-13 and float(hessian_error)<1e-25
            records.append(dict(r=r,I=I,e=e.tolist(),Hpp=H.tolist(),g=g.tolist(),recovered=base.tolist(),
                euclidean_error=error,physical_hessian_check=float(hessian_error),
                radial_hvp_identity_error=float(np.linalg.norm(v-I*I*a*J@H@e-I*K@g[1:]))))
            if r not in [.9,1.,1.1]:continue
            B=np.array([[.6,.3],[.3,-.4]]);B/=np.linalg.norm(B,2)
            for epsH in [0.,1e-8,1e-5,1e-2]:
                dH=epsH*B;dh=dH@e
                curvature_only=recover(b,v,(H+dH)@e)
                predicted=C@dh
                curvature_delta=curvature_only-base
                epsb=epsv=1e-6
                mixed=recover(b-.75*epsb,v+.8*epsv*e,(H+dH)@e)
                bound=epsb+np.linalg.norm(R,2)*epsv+np.linalg.norm(C,2)*epsH
                mixed_error=float(np.linalg.norm(mixed-g))
                residual=float(np.linalg.norm(curvature_delta-predicted))
                assert residual<2e-13 and mixed_error<=bound+2e-13
                if r==1.:assert np.array_equal(curvature_delta,np.zeros(3))
                bias.append(dict(r=r,eps_H=epsH,delta_H=dH.tolist(),eps_b=epsb,eps_v=epsv,
                    curvature_error=float(np.linalg.norm(curvature_delta)),curvature_error_bound=float(np.linalg.norm(C,2)*epsH),
                    curvature_identity_residual=residual,total_error=mixed_error,total_bound=float(bound)))
    report=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='One quartic objective; seven radius checks and 12 supplied curvature-input perturbations. Geometry and reporting norm exact; no calibration/radius errors propagated.',
        reporting_norm='Euclidean (M=identity), not the local physical Hessian',
        recovery_records=records,curvature_error_records=bias,
        summary=dict(recovery_cases=len(records),curvature_error_cases=len(bias),
            max_recovery_error=max(t['euclidean_error'] for t in records),
            max_physical_hessian_check=max(t['physical_hessian_check'] for t in records),
            max_radial_hvp_identity_error=max(t['radial_hvp_identity_error'] for t in records),
            max_curvature_identity_residual=max(t['curvature_identity_residual'] for t in bias),
            max_fold_curvature_error=max(t['curvature_error'] for t in bias if t['r']==1.)))
    (HERE/'results/local_curvature_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
