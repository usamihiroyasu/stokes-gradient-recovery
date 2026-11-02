"""Information-identifiability checks and primal-dual audits of retained cone QPs.

No new training or simulation series. All perturbation envelopes are prescribed.
Primal-dual inequalities are exact-arithmetic statements; float64 evaluation is
recorded as numerical evidence, not an interval-arithmetic certification.
"""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'

def soc(x):
    x=np.asarray(x,dtype=float);t=x[0];p=x[1:];r=np.linalg.norm(p)
    if r<=t:y=x.copy()
    elif r<=-t:y=np.zeros_like(x)
    else:
        a=.5*(t+r);y=np.r_[a,a*p/r]
    # Repair a negative roundoff slack before treating a vector as cone-feasible.
    slack=y[0]-np.linalg.norm(y[1:])
    if slack<0:y[0]+=-slack+8*np.finfo(float).eps*max(1.,abs(y[0]))
    return y

def bracket(W,z,x,s):
    u=soc(x);g=W@(u-z)
    candidates=[np.zeros(3),soc(g)]
    polar_norm=np.linalg.norm(u[1:])
    if polar_norm>1e-14:
        direction=soc(np.r_[1.,-u[1:]/polar_norm])
        gamma=max(0.,-float(direction@z)/float(direction@np.linalg.solve(W,direction)))
        candidates.append(soc(gamma*direction))
    def dual(q):return float(-q@z-.5*q@np.linalg.solve(W,q))
    q=max(candidates,key=dual)
    # Select among explicitly cone-feasible dual candidates, including the
    # optimal scalar coefficient on the candidate primal supporting-normal ray.
    P=float(.5*(u-z)@W@(u-z));D=dual(q)
    Ls=float(.5*(s-z)@W@(s-z));width=P-D
    assert width>=-2e-12
    assert u[0]>=np.linalg.norm(u[1:]) and q[0]>=np.linalg.norm(q[1:])
    return dict(primal=P,dual=D,width=width,gap_lower=Ls-P,gap_upper=Ls-D,
                primal_repair=float(np.linalg.norm(u-x)),u=u.tolist(),q=q.tolist())

def raw(I,r,e,z,W):
    phi=2*r/(1+r*r);a=2*(1-r*r)/(1+r*r)**2;t=2/(1+r*r)
    P=np.outer(e,e);J=a*P+t*(np.eye(2)-P)
    K=4*r*(r*r-3)/(1+r*r)**3*P-4*r/(1+r*r)**2*(np.eye(2)-P)
    s=I*np.r_[1.,phi*e];g=W@(s-z)
    b=float(g[0]+phi*e@g[1:]);j=I*J@g[1:]
    v=I*I*a*J@W[1:,1:]@e+I*K@g[1:]
    return phi,a,t,J,K,g,b,j,v

def main():
    old84=json.loads((RESULTS/'general_metric_independent.json').read_text())
    old108=json.loads((RESULTS/'fold_gradient_recovery_independent.json').read_text())
    old82=json.loads((RESULTS/'robust_sensitivity.json').read_text())
    audit=[];scatter=[]
    for tag,data in [('stationary84',old84),('nonstationary108',old108),('sensitivity82',old82)]:
        matrices={m['name']:np.array(m['W']) for m in data.get('matrices',[])}
        order={m['name']:i for i,m in enumerate(data.get('matrices',[]))}
        for i,row in enumerate(data['records']):
            W=np.array(row['W']) if 'W' in row else matrices[row['matrix']]
            z=np.array(row['z']);I=row['I']
            if tag=='sensitivity82':
                e=np.array([np.cos(.7),np.sin(.7)]);s=I*np.r_[1.,row['phi']*e]
                x=np.array(row['qp']['x']);bound=row['robust_bound']
            else:
                e=np.array(row['e']);s=I*np.r_[1.,e];x=np.array(row['qp']['solution'])
                bound=row['certificate_bound'] if tag=='stationary84' else row['certificate_bound_exact']
            a=bracket(W,z,x,s);a.update(block=tag,index=i,bound=bound,bound_minus_gap_upper=bound-a['gap_upper'])
            assert a['bound_minus_gap_upper']>=-1e-10
            audit.append(a)
            if tag=='stationary84' and row['alpha']>0:
                scatter.append(dict(bound=bound,computed_gap=a['gap_lower'],z_feasible=row['z_feasible'],matrix_index=order[row['matrix']]))
    summaries={tag:dict(cases=sum(a['block']==tag for a in audit),
                        max_primal_dual_width=max(a['width'] for a in audit if a['block']==tag),
                        min_bound_minus_gap_upper=min(a['bound_minus_gap_upper'] for a in audit if a['block']==tag),
                        max_primal_repair=max(a['primal_repair'] for a in audit if a['block']==tag))
               for tag in ['stationary84','nonstationary108','sensitivity82']}
    # Figure 1: same value, same first head derivatives, same physical Hessian.
    I=1.;e=np.array([1.,0.]);s=np.r_[1.,e];W=np.diag([1.,.5,.5]);n=np.r_[-1.,e]
    ts=np.linspace(-.35,.35,351);cases=[]
    for alpha,label in [(-.3,'constrained optimum'),(0.,'constrained optimum'),(.3,'strict saddle')]:
        z=s-alpha*np.linalg.solve(W,n);_,_,_,_,_,g,b,j,v=raw(I,1.,e,z,W)
        assert np.linalg.norm(np.r_[b,j])<1e-14
        cr=float(e@v);assert abs(cr+I*alpha)<1e-14
        vals=[]
        for t0 in ts:
            rr=1+t0;phi=2*rr/(1+rr*rr);ss=I*np.r_[1.,phi*e]
            vals.append(float(.5*(ss-z)@W@(ss-z)-.5*(s-z)@W@(s-z)))
        cases.append(dict(alpha=alpha,z=z.tolist(),g=g.tolist(),curvature=cr,radial_loss=vals,classification=label))
    # Figure 3: fixed controlled derivative errors, on both radius sheets.
    W=np.array(old84['matrices'][1]['W']);M=np.linalg.inv(W)
    I=1.2;e=np.array([np.cos(.7),np.sin(.7)]);z=np.array([.2,1.1,-.4]);P=np.outer(e,e)
    epsb=epsj=epsv=1e-6;compare=[]
    for delta in np.logspace(-7,-1,15):
        for sign in [-1,1]:
            r=1+sign*delta;phi,a,t,J,K,g,b,j,v=raw(I,r,e,z,W)
            b0=b-.75*epsb;j0=j+.8*epsj*e;v0=v+.8*epsv*e
            Jinv=P/a+(np.eye(2)-P)/t
            gp=Jinv@j0/I;first=np.r_[b0-phi*e@gp,gp]
            second=np.r_[b0+e@v0/I,-v0/I]
            first_exact_p=Jinv@j/I;first_exact=np.r_[b-phi*e@first_exact_p,first_exact_p]
            kb=np.sqrt(M[0,0]);T=np.vstack([-phi*e,np.eye(2)])@Jinv/I
            eta1=kb*epsb+np.sqrt(np.linalg.eigvalsh(T.T@M@T)[-1])*epsj
            S=np.vstack([e,-np.eye(2)]);kv=np.sqrt(np.linalg.eigvalsh(S.T@M@S)[-1])
            G=I*(np.linalg.norm(W[1:,0])+np.linalg.norm(W[1:,1:],2))+np.linalg.norm((W@z)[1:])
            eta2=kb*((1-phi)*G+epsb)+kv*(I*abs(a)*np.linalg.norm(J,2)*np.linalg.norm(W[1:,1:],2)+np.linalg.norm(K+np.eye(2),2)*G+epsv/I)
            norm=lambda x:float(np.sqrt(max(0.,x@M@x)))
            row=dict(delta=float(delta),sheet='inner' if sign<0 else 'outer',first_error=norm(first-g),first_envelope=float(eta1),second_error=norm(second-g),second_envelope=float(eta2),exact_first_error=norm(first_exact-g),r=float(r),epsb=epsb,epsj=epsj,epsv=epsv)
            assert row['first_error']<=eta1+1e-9 and row['second_error']<=eta2+1e-10
            assert row['exact_first_error']<1e-7
            compare.append(row)
    # Exact-gradient support-slack counterexample, independent analytic projection.
    r=.8;I=1.;e=np.array([1.,0.]);W=np.diag([1.,.5,.5]);M=np.linalg.inv(W);n=np.r_[-1.,e]
    phi=2*r/(1+r*r);s=I*np.r_[1.,phi*e];sigma=I*(1-phi);z=s+.2*M@n
    _,_,_,_,_,g,b,j,v=raw(I,r,e,z,W);ghat=np.r_[b+e@v/I,-v/I]
    opt=np.array([122/123,122/123,0]);gap=float(.5*(s-z)@W@(s-z)-.5*(opt-z)@W@(opt-z))
    lam0=max(0.,-float(n@M@g)/float(n@M@n));lam=max(0.,-float(n@M@g+sigma)/float(n@M@n))
    unc=.5*(g+lam0*n)@M@(g+lam0*n);corrected=.5*(g+lam*n)@M@(g+lam*n)+lam*sigma
    assert abs(gap-241/50430)<1e-14 and abs(corrected-gap)<1e-14 and abs(unc)<1e-28
    report=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Existing records audited; 30 matched prescribed-envelope radius points; no new optimization/training series',
                numerical_status='Float64 primal-dual values, not outward-rounded interval enclosures',
                derivative_comparison='Exact radius/intensity/metric; epsb=epsj=epsv=1e-6; errors -0.75 epsb,+0.8 epsj e,+0.8 epsv e; equal numerical envelopes do not establish equal cost or universal ranking',
                bracket_summary=summaries,primal_dual_records=audit,radius_records=compare,
                exact_first_max_error=max(row['exact_first_error'] for row in compare),
                figure1=dict(I=1.,e=[1.,0.],s=[1.,1.,0.],W=np.diag([1.,.5,.5]).tolist(),t=ts.tolist(),cases=cases),
                figure2=dict(pI_s=[.8,.8],pI_h=[-1.4,.2],pI_opt=[-.6,.6],W_pI=[[.5,0],[0,1]],values=dict(current=1.39,cone_minimum=.24,halfspace_minimum=0),scatter=scatter),
                figure3=dict(bias=dict(g=g.tolist(),ghat=ghat.tolist(),radius=r),support=dict(radius=r,I=I,phi=phi,sigma=sigma),comparison=compare,counterexample=dict(uncorrected_bound=float(unc),exact_gap=gap,corrected_bound=float(corrected))))
    (RESULTS/'identifiability_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],brackets=summaries,exact_first_max_error=report['exact_first_max_error'],radius_cases=len(compare),counterexample=report['figure3']['counterexample']),indent=2))
if __name__=='__main__':main()
