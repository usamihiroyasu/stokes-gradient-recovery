"""Known-metric recovery and nonquadratic fold checks; no fitted parameters.

Decimal differences independently evaluate composite objectives/gradients.
Their discrepancies are checks, not certified derivative-error envelopes.
"""
from pathlib import Path
from decimal import Decimal as D, localcontext
import hashlib, json
import numpy as np
from identifiability_checks import raw

HERE = Path(__file__).resolve().parent

def dvec(x):
    return [D(str(float(v))) for v in x]

def dot(x, y):
    return sum(a*b for a,b in zip(x,y))

def loss_grad(s, kind):
    x,y,z=s
    if kind == 'quartic':
        q=y+D('.4')*z
        f=D('.2')*x**4+D('.1')*q**4+D('.3')*x*y*z+D('.05')*z**3+D('.4')*x+D('.2')*y-D('.3')*z
        g=[D('.8')*x**3+D('.3')*y*z+D('.4'),D('.4')*q**3+D('.3')*x*z+D('.2'),D('.16')*q**3+D('.3')*x*y+D('.15')*z*z-D('.3')]
    elif kind == 'exponential_quartic':
        c=[D('.3'),D('-.2'),D('.5')];ex=dot(c,s).exp();q=dot(s,s)
        f=ex+D('.03')*q*q
        g=[ex*c[i]+D('.12')*q*s[i] for i in range(3)]
    elif kind == 'softplus_coupled':
        c=[D('-.4'),D('.7'),D('.2')];ex=dot(c,s).exp()
        f=(1+ex).ln()+D('.07')*x*y*z+D('.05')*dot(s,s)
        extra=[y*z,x*z,x*y]
        g=[ex/(1+ex)*c[i]+D('.07')*extra[i]+D('.1')*s[i] for i in range(3)]
    else:
        raise ValueError(kind)
    return f,g

def head_values(intensity,w,physical):
    den=1+dot(w,w);F=[2*x/den for x in w]
    s=[intensity]+[intensity*x for x in F];f,g=physical(s)
    J=[[2*D(int(i==j))/den-4*w[i]*w[j]/den**2 for j in range(2)] for i in range(2)]
    b=g[0]+dot(F,g[1:]);j=[intensity*dot(J[i],g[1:]) for i in range(2)]
    return f,b,j,g

def main():
    original=json.loads((HERE/'results/identifiability_checks.json').read_text())
    prior=json.loads((HERE/'results/general_metric_independent.json').read_text())
    W=np.array(prior['matrices'][1]['W']);M=np.linalg.inv(W)
    intensity=1.2;e=np.array([np.cos(.7),np.sin(.7)]);z=np.array([.2,1.1,-.4]);P=np.outer(e,e)
    eps=1e-6;radius=[]
    with localcontext() as ctx:
        ctx.prec=80
        for old in original['radius_records']:
            r=old['r'];phi,a,t,J,K,g,b,j,v=raw(intensity,r,e,z,W)
            ki=np.linalg.inv(K);A=np.vstack([-phi*e,np.eye(2)]);R=A@ki/intensity
            correction=intensity*a*J@W[1:,1:]@e
            recover=lambda bb,vv: np.r_[bb-phi*e@(ki@(vv/intensity-correction)),ki@(vv/intensity-correction)]
            exact=recover(b,v);estimate=recover(b-.75*eps,v+.8*eps*e)
            norm=lambda x:float(np.sqrt(max(0.,x@M@x)))
            envelope=np.sqrt(M[0,0])*eps+np.sqrt(np.linalg.eigvalsh(R.T@M@R)[-1])*eps
            # Independent radial differentiation of the raw gradient in Decimal.
            wd=[dvec(W[i]) for i in range(3)];zd=dvec(z)
            def quadratic(s):
                diff=[s[i]-zd[i] for i in range(3)];gg=[dot(row,diff) for row in wd]
                return dot(diff,gg)/2,gg
            h=D('1e-18');ed=dvec(e);rd=D(str(r));Id=D(str(intensity))
            jm=head_values(Id,[(rd-h)*q for q in ed],quadratic)[2]
            jp=head_values(Id,[(rd+h)*q for q in ed],quadratic)[2]
            vfd=np.array([float((jp[i]-jm[i])/(2*h)) for i in range(2)])
            row=dict(old,corrected_error=norm(estimate-g),corrected_envelope=float(envelope),
                     corrected_exact_error=norm(exact-g),corrected_hvp_decimal_error=float(np.linalg.norm(vfd-v)),
                     K_eigenvalues=np.linalg.eigvalsh(K).tolist())
            assert row['corrected_error']<=envelope+1e-12
            assert row['corrected_exact_error']<1e-12 and row['corrected_hvp_decimal_error']<1e-12
            radius.append(row)

        nonlinear=[];h=D('1e-18');h2=D('1e-12')
        for kind in ['quartic','exponential_quartic','softplus_coupled']:
            for intensity0 in [.2,1.2,5.]:
                for angle in [0.,.7,1.9]:
                    ed=dvec([np.cos(angle),np.sin(angle)])
                    length=dot(ed,ed).sqrt();ed=[x/length for x in ed]
                    Id=D(str(intensity0));physical=lambda s:loss_grad(s,kind)
                    f,b,j,g=head_values(Id,ed,physical)
                    fm=head_values(Id,[(1-h2)*q for q in ed],physical)[0]
                    fp=head_values(Id,[(1+h2)*q for q in ed],physical)[0]
                    cr=(fp-2*f+fm)/(h2*h2)
                    gp=[(j[i]-cr*ed[i])/Id for i in range(2)]
                    scalar=[b+cr/Id]+gp
                    jm=head_values(Id,[(1-h)*q for q in ed],physical)[2]
                    jp=head_values(Id,[(1+h)*q for q in ed],physical)[2]
                    v=[(jp[i]-jm[i])/(2*h) for i in range(2)]
                    vector=[b+dot(ed,v)/Id]+[-q/Id for q in v]
                    s=[Id]+[Id*q for q in ed];physical_fd=[]
                    for k in range(3):
                        sm=s.copy();sp=s.copy();sm[k]-=h;sp[k]+=h
                        physical_fd.append((physical(sp)[0]-physical(sm)[0])/(2*h))
                    ef=np.array([float(q) for q in ed]);Df=np.column_stack([np.r_[1.,ef],np.vstack([np.zeros(2),intensity0*(np.eye(2)-np.outer(ef,ef))])])
                    augmented=np.vstack([Df.T,np.r_[0.,-intensity0*ef]])
                    row=dict(objective=kind,I=intensity0,angle=angle,
                        physical_gradient=[float(x) for x in g],raw_gradient_norm=float(np.linalg.norm([float(b),*[float(q) for q in j]])),
                        scalar_recovery_error=float(max(abs(scalar[i]-g[i]) for i in range(3))),
                        vector_recovery_error=float(max(abs(vector[i]-g[i]) for i in range(3))),
                        physical_derivative_check=float(max(abs(physical_fd[i]-g[i]) for i in range(3))),
                        first_rank=int(np.linalg.matrix_rank(Df.T)),augmented_rank=int(np.linalg.matrix_rank(augmented)))
                    assert row['scalar_recovery_error']<1e-18 and row['vector_recovery_error']<1e-18
                    assert row['physical_derivative_check']<1e-25
                    assert row['first_rank']==2 and row['augmented_rank']==3
                    nonlinear.append(row)
    report=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assumptions='Known exact I,e,r,Iprime,W; prescribed derivative envelopes. Decimal differences are consistency checks, not outward-rounded certificates.',
        radius_records=radius,nonquadratic_records=nonlinear,
        summary=dict(radius_cases=len(radius),nonquadratic_cases=len(nonlinear),
            max_corrected_exact_error=max(q['corrected_exact_error'] for q in radius),
            max_corrected_hvp_decimal_error=max(q['corrected_hvp_decimal_error'] for q in radius),
            corrected_error_range=[min(q['corrected_error'] for q in radius),max(q['corrected_error'] for q in radius)],
            corrected_envelope_range=[min(q['corrected_envelope'] for q in radius),max(q['corrected_envelope'] for q in radius)],
            max_nonlinear_scalar_error=max(q['scalar_recovery_error'] for q in nonlinear),
            max_nonlinear_vector_error=max(q['vector_recovery_error'] for q in nonlinear),
            max_physical_derivative_discrepancy=max(q['physical_derivative_check'] for q in nonlinear),
            min_abs_K_eigenvalue=min(min(abs(x) for x in q['K_eigenvalues']) for q in radius)))
    (HERE/'results/corrected_recovery_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
