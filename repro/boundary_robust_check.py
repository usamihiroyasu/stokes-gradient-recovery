"""Deterministic near-fold formulas and analytic ideal-cone gap checks."""
import json
from pathlib import Path
import numpy as np

W=np.diag([1.,.5,.5]); M=np.linalg.inv(W)
I=.7; angle=.71; e=np.array([np.cos(angle),np.sin(angle)])
e0=np.array([1.,0.,0.]); S=np.vstack([e,-np.eye(2)])
n=np.r_[-1.,e]; D=n@M@n
errs=dict(hvp=0.,bias=0.,bias_envelope_excess=0.,robust_gap_excess=0.)
cases=0
def F(w): return 2*w/(1+w@w)
def JF(w):
    d=1+w@w
    return 2*np.eye(2)/d-4*np.outer(w,w)/d**2
def project(z):
    t,p=z[0],z[1:]; rho=np.linalg.norm(p)
    if t>=rho: return z
    b=max(0.,(2*t+rho)/3)
    return np.r_[b,b*p/rho] if rho else np.array([max(t,0.),0.,0.])
for r in [.6,.9,.99,1.,1.01,1.2,1.8]:
    phi=2*r/(1+r*r); psi=2/(1+r*r)
    a=2*(1-r*r)/(1+r*r)**2
    phipp=4*r*(r*r-3)/(1+r*r)**3
    psip=-4*r/(1+r*r)**2
    J=JF(r*e); K=phipp*np.outer(e,e)+psip*(np.eye(2)-np.outer(e,e))
    for z in [np.array(x) for x in [(1.,.2,.1),(-.2,-.9,.7),(.4,1.6,-.8),(.7,.2,-.3)]]:
        state=np.r_[I,I*phi*e]; g=W@(state-z); gp=g[1:]
        def rawgrad(w): return I*JF(w)@(W@(np.r_[I,I*F(w)]-z))[1:]
        h=1e-5
        vfd=(rawgrad((r+h)*e)-rawgrad((r-h)*e))/(2*h)
        v=I*I*a*J@W[1:,1:]@e+I*K@gp
        errs['hvp']=max(errs['hvp'],float(np.max(abs(v-vfd))))
        b=g[0]+phi*(e@gp); naive=e0*b+S@v/I
        kr=I*a*S@J@W[1:,1:]@e
        Ar=(phi-1)*np.outer(e0,e)+S@(K+np.eye(2))
        bias=kr+Ar@gp
        errs['bias']=max(errs['bias'],float(np.max(abs(naive-g-bias))))
        eta=np.sqrt(kr@M@kr)+np.sqrt(np.linalg.eigvalsh(Ar.T@M@Ar)[-1])*np.linalg.norm(gp)
        errs['bias_envelope_excess']=max(errs['bias_envelope_excess'],float(np.sqrt(bias@M@bias)-eta))
        db=.004; dv=np.array([-.01,.005]); estimate=e0*(b+db)+S@(v+dv)/I
        eta+=np.sqrt(M[0,0])*abs(db)+np.sqrt(np.linalg.eigvalsh(S.T@M@S)[-1])*np.linalg.norm(dv)/I
        sigma=I*(1-phi); lam=max(0.,-(n@M@estimate+sigma)/D)
        vec=estimate+lam*n
        bound=.5*(np.sqrt(vec@M@vec)+eta)**2+lam*sigma
        pc=project(z)
        gap=.5*(state-z)@W@(state-z)-.5*(pc-z)@W@(pc-z)
        errs['robust_gap_excess']=max(errs['robust_gap_excess'],float(gap-bound))
        cases+=1

# Exact counterexample to using the zero-slack formula away from the fold.
r=.8; phi=2*r/(1+r*r); state=np.r_[1.,phi*e]
z=state+.2*M@n; pc=project(z)
counter_gap=.5*(state-z)@W@(state-z)-.5*(pc-z)@W@(pc-z)
report=dict(scope='Independent deterministic near-fold derivative and analytic ideal-cone projection checks',
            numpy_version=np.__version__,cases=cases,finite_difference_step=h,
            max_errors=errs,zero_slack_counterexample=dict(radius=r,old_bound=0.,true_gap=float(counter_gap),exact_gap_fraction='241/50430'))
assert errs['hvp']<1e-7
assert errs['bias']<1e-12
assert errs['bias_envelope_excess']<1e-12
assert errs['robust_gap_excess']<1e-12
assert abs(counter_gap-241/50430)<1e-12
(Path(__file__).resolve().parent/'results/boundary_robust_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
