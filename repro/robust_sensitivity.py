"""Controlled sensitivity of the near-fold certificate; no retained data fitting.

All derivative perturbations obey supplied deterministic envelopes. G is an
analytic a-priori triangle bound from the fixed quadratic, never observed error.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'mpl'))


def cone_slack(x):return float(x[0]-np.linalg.norm(x[1:]))


def solve_qp(z,W):
    g0=-W@z
    # Lorentz cone is self-dual: apex optimal iff g(0) belongs to C.
    apex_dual_slack=cone_slack(g0)
    if apex_dual_slack>=0:
        return dict(status=0,success=True,mode='exact_apex_self_dual_test',x=[0.,0.,0.],
                    objective=float(.5*z@W@z),cone_violation=0.,stationarity=0.,
                    complementarity=0.,apex_dual_slack=apex_dual_slack)
    def f(x):return float(.5*(x-z)@W@(x-z))
    def jac_h(x):
        qnorm=np.linalg.norm(x[1:])
        if qnorm<1e-12:
            # A valid concave supergradient is used for the solver only; it is
            # never used to declare apex KKT. All returned nonapex points below
            # must have nonzero polar norm and are checked with their true normal.
            return np.array([1.,0.,0.])
        return np.r_[1.,-x[1:]/qnorm]
    res=minimize(f,[1.5,.1,-.1],jac=lambda x:W@(x-z),method='SLSQP',
                 constraints=[dict(type='ineq',fun=cone_slack,jac=jac_h)],
                 options=dict(ftol=1e-13,maxiter=2000))
    x=res.x;h=cone_slack(x);g=W@(x-z);pnorm=float(np.linalg.norm(x[1:]))
    if np.linalg.norm(x)<1e-8:
        # Never certify a nonsmooth apex using the single-normal fallback.
        raise AssertionError('Unexpected apex from solver although exact self-dual test failed')
    if h<1e-7:
        assert pnorm>1e-9,'nonapex normal requires nonzero polar norm'
        normal=np.r_[-1.,x[1:]/pnorm]
        lam=max(0.,-float(g@normal)/float(normal@normal))
    else:normal=np.zeros(3);lam=0.
    return dict(status=int(res.status),success=bool(res.success),mode='SLSQP_nonapex',x=x.tolist(),
                objective=f(x),cone_violation=max(0.,-h),stationarity=float(np.linalg.norm(g+lam*normal,np.inf)),
                complementarity=abs(lam*h),polar_norm=pnorm,solution_norm=float(np.linalg.norm(x)),
                normal_applied=bool(h<1e-7),apex_dual_slack=apex_dual_slack)


def raw(intensity,w,z,W):
    den=1+w@w;F=2*w/den;J=2*np.eye(2)/den-4*np.outer(w,w)/den**2
    s=intensity*np.r_[1.,F];g=W@(s-z)
    return float(.5*(s-z)@g),float(np.r_[1.,F]@g),intensity*J@g[1:]


def metric():
    theta=np.deg2rad([0,37,91,143]);gain=np.array([1.03,.97,1.01,.99]);eff=.94
    A=.5*gain[:,None]*np.column_stack([np.ones(4),eff*np.cos(2*theta),eff*np.sin(2*theta)])
    W=A.T@(A/np.array([1,.8,1.1,1.3])[:,None]);return 2*W/np.trace(W)


def make_derivative_interface(W,z,linear=None,anchor=None):
    """Constructed wrapper, not evidence that a real API hides physical gradients.

    Interface(raw=(I,w)) -> value, rawgrad, radial HVP v, b, W only.
    This transparent known quadratic permits direct g computation in principle;
    the restricted interface is imposed solely to demonstrate the algebra.
    """
    def interface(raw_coordinates):
        I=float(raw_coordinates[0]);w=np.asarray(raw_coordinates[1:]);r=float(np.linalg.norm(w))
        assert I>0 and r>0
        e=w/r;P=np.outer(e,e);T=np.eye(2)-P
        phi=2*r/(1+r*r);a=2*(1-r*r)/(1+r*r)**2;psi=2/(1+r*r)
        c=4*r*(r*r-3)/(1+r*r)**3;d=-4*r/(1+r*r)**2
        J=a*P+psi*T;DJ=c*P+d*T
        value,b,gw=raw(I,w,z,W)
        s=I*np.r_[1.,phi*e]
        if linear is not None:
            value+=float(linear@(s-anchor))
            b+=float(np.r_[1.,phi*e]@linear)
            gw+=I*J@linear[1:]
        gp=(W@(s-z)+(np.zeros(3) if linear is None else linear))[1:]
        v=I*I*a*J@W[1:,1:]@e+I*DJ@gp
        return dict(value=value,rawgrad=np.r_[b,gw],v=v,b=b,W=W.copy())
    return interface


def _private_validation_oracle(I,phi,e,W,z):
    """Physical gradient exposed exclusively to validation, not wrapper output."""
    return W@(I*np.r_[1.,phi*e]-z)


def dual_norm(g,Winv):return float(np.sqrt(max(0.,g@Winv@g)))


def evaluate(label,x,r,I,W,z,epsb,epsv,sheet='none'):
    e=np.array([np.cos(.7),np.sin(.7)]);P=np.outer(e,e);T=np.eye(2)-P;n=np.r_[-1.,e]
    phi=2*r/(1+r*r);a=2*(1-r*r)/(1+r*r)**2;psi=2/(1+r*r)
    c=4*r*(r*r-3)/(1+r*r)**3;d=-4*r/(1+r*r)**2
    J=a*P+psi*T;DJ=c*P+d*T;Wpp=W[1:,1:];s=I*np.r_[1.,phi*e]
    derivative_interface=make_derivative_interface(W,z)
    exposed=derivative_interface(np.r_[I,r*e])
    assert set(exposed)=={'value','rawgrad','v','b','W'}
    g=_private_validation_oracle(I,phi,e,W,z);gp=g[1:]
    b=exposed['b'];v=exposed['v']
    h=2e-5;vfd=(raw(I,(r+h)*e,z,W)[2]-raw(I,(r-h)*e,z,W)[2])/(2*h)
    # Supplied errors are imposed before looking at reconstruction error.
    delta_b=-.75*epsb;delta_v=.8*epsv*e
    bhat=b+delta_b;vhat=v+delta_v
    ghat=np.r_[bhat+float(e@vhat)/I,-vhat/I]
    Winv=np.linalg.inv(W);S=np.vstack([e,-np.eye(2)])
    kb=float(np.sqrt(Winv[0,0]));kv=float(np.sqrt(np.linalg.eigvalsh(S.T@Winv@S).max()))
    # Since ||I phi e|| <= I, this holds a priori for every radius and direction.
    G=float(I*(np.linalg.norm(W[1:,0])+np.linalg.norm(Wpp,2))+np.linalg.norm(W[1:,:]@z))
    geometric_b=max(0.,1-phi)*G
    geometric_v=I*abs(a)*np.linalg.norm(J,2)*np.linalg.norm(Wpp,2)+np.linalg.norm(DJ+np.eye(2),2)*G
    eta=kb*(geometric_b+epsb)+kv*(geometric_v+epsv/I)
    actual_error=dual_norm(ghat-g,Winv)
    lam=max(0.,-float(n@Winv@ghat)/float(n@Winv@n))
    residual_norm=dual_norm(ghat+lam*n,Winv);sigma=max(0.,I*(1-phi))
    robust=.5*(residual_norm+eta)**2+lam*sigma
    naive=.5*residual_norm**2
    qp=solve_qp(z,W);gap=float(.5*(s-z)@W@(s-z)-qp['objective'])
    return dict(panel=label,x=float(x),sheet=sheet,r=float(r),I=float(I),W=W.tolist(),z=z.tolist(),
                epsb=float(epsb),epsv=float(epsv),G=G,actual_gp_norm=float(np.linalg.norm(gp)),
                delta_b=delta_b,delta_v_norm=float(np.linalg.norm(delta_v)),
                phi=phi,a=a,psi=psi,c=c,d=d,g=g.tolist(),ghat=ghat.tolist(),
                analytic_HVP=v.tolist(),FD_HVP=vfd.tolist(),
                HVP_FD_error=float(np.max(abs(v-vfd))),
                HVP_FD_scaled_error=float(np.max(abs(v-vfd))/(1+np.max(abs(v)))),
                kb=kb,kv=kv,eta=eta,actual_error=actual_error,
                error_safe_margin=eta-actual_error,lambda_value=lam,sigma=sigma,
                naive_bound=naive,robust_bound=robust,actual_gap=gap,
                gap_safe_margin=robust-gap,qp=qp)


def main():
    W=metric();z=np.array([.2,1.1,-.4]);rows=[]
    for delta in np.logspace(-7,-1,15):
        for sign in [-1,1]:rows.append(evaluate('radius',delta,1+sign*delta,1.2,W,z,0.,0.,'inner' if sign<0 else 'outer'))
    rows.append(evaluate('exact_fold',0.,1.,1.2,W,z,0.,0.))
    for eps in np.logspace(-8,-1,17):rows.append(evaluate('derivative',eps,1.,1.2,W,np.array([.8,0,0]),eps,eps))
    for intensity in np.logspace(-3,1,17):rows.append(evaluate('intensity',intensity,1.,intensity,W,z,1e-6,1e-5))
    # Fixed eigenbasis, maximum eigenvalue=1; metric conditioning only changes.
    ca,sa=np.cos(.4),np.sin(.4);cb,sb=np.cos(.6),np.sin(.6)
    R=np.array([[ca,-sa,0],[sa,ca,0],[0,0,1.]])@np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]])
    for kappa in np.logspace(0,4,17):
        Wk=R@np.diag([1.,1.,1./kappa])@R.T
        rows.append(evaluate('metric',kappa,1.,1.2,Wk,z,1e-6,1e-5))
    apexchecks=[]
    for target in [np.array([1.,0.,0.]),np.array([1.5,.2,-.3]),np.array([2.,-.5,.4])]:
        za=-np.linalg.solve(W,target);qp=solve_qp(za,W)
        assert qp['mode']=='exact_apex_self_dual_test'
        apexchecks.append(dict(z=za.tolist(),g0=target.tolist(),qp=qp))
    # Separate pointwise tests; do not count these as sensitivity cases.
    e=np.array([np.cos(.7),np.sin(.7)]);n=np.r_[-1.,e];I=1.2
    anchor=I*np.r_[1.,e];zi=np.array([.8,0.,0.]);auxiliary=[]
    for tau in [-1.,0.,1.]:
        out=make_derivative_interface(W,zi,tau*n,anchor)(np.r_[I,e])
        physical_gradient=W@(anchor-zi)+tau*n
        cr=float(e@out['v'])
        scalar_reconstruction=np.r_[out['b']+cr/I,(out['rawgrad'][1:]-cr*e)/I]
        auxiliary.append(dict(tau=tau,value=out['value'],raw_gradient=out['rawgrad'].tolist(),
                              physical_gradient=physical_gradient.tolist(),physical_Hessian=W.tolist(),
                              radial_curvature=cr,raw_gradient_norm=float(np.linalg.norm(out['rawgrad'])),
                              scalar_reconstruction_error=float(np.max(abs(scalar_reconstruction-physical_gradient)))))
    tau0=auxiliary[1]
    auxiliary_tests=dict(
        first_order_values_identical=max(abs(t['value']-tau0['value']) for t in auxiliary)<1e-14,
        first_order_raw_gradients_identical=max(np.max(abs(np.array(t['raw_gradient'])-tau0['raw_gradient'])) for t in auxiliary)<1e-14,
        physical_gradient_shift_is_tau_n=max(np.max(abs(np.array(t['physical_gradient'])-tau0['physical_gradient']-t['tau']*n)) for t in auxiliary)<1e-14,
        scalar_curvature_reconstructs_full_gradient=max(t['scalar_reconstruction_error'] for t in auxiliary)<1e-12,
        all_auxiliary_points_nonstationary=min(t['raw_gradient_norm'] for t in auxiliary)>1e-6,
        curvature_shift_is_minus_I_tau=max(abs(t['radial_curvature']-tau0['radial_curvature']+I*t['tau']) for t in auxiliary)<1e-12,
    )
    auxiliary_tests={key:bool(value) for key,value in auxiliary_tests.items()}
    tests=dict(
        supplied_G_valid=all(r['actual_gp_norm']<=r['G']+1e-12 for r in rows),
        controlled_derivative_errors_obey_envelopes=all(abs(r['delta_b'])<=r['epsb']+1e-15 and r['delta_v_norm']<=r['epsv']+1e-15 for r in rows),
        gradient_error_obeys_envelope=all(r['actual_error']<=r['eta']+1e-11 for r in rows),
        actual_gap_obeys_robust_bound=all(r['actual_gap']<=r['robust_bound']+1e-10 for r in rows),
        independent_FD_HVP=all(r['HVP_FD_scaled_error']<1e-7 for r in rows),
        solver_success=all(r['qp']['success'] for r in rows),
        solver_feasible=all(r['qp']['cone_violation']<1e-8 for r in rows),
        solver_KKT=all(r['qp']['stationarity']<3e-6 for r in rows),
        solver_complementarity=all(r['qp']['complementarity']<1e-8 for r in rows),
        apex_self_dual_checks=all(r['qp']['apex_dual_slack']>=0 for r in apexchecks),
        nonapex_normals_valid=all(r['qp'].get('polar_norm',1.)>1e-9 for r in rows if r['qp']['mode']=='SLSQP_nonapex'),
    )
    summary=dict(cases=len(rows),panels={p:sum(r['panel']==p for r in rows) for p in ['radius','exact_fold','derivative','intensity','metric']},
                 separate_apex_checks=len(apexchecks),
                 max_HVP_FD_error=max(r['HVP_FD_error'] for r in rows),
                 max_HVP_FD_scaled_error=max(r['HVP_FD_scaled_error'] for r in rows),
                 max_solver_stationarity=max(r['qp']['stationarity'] for r in rows),
                 max_solver_cone_violation=max(r['qp']['cone_violation'] for r in rows),
                 max_solver_complementarity=max(r['qp']['complementarity'] for r in rows),
                 min_error_safe_margin=min(r['error_safe_margin'] for r in rows),
                 min_gap_safe_margin=min(r['gap_safe_margin'] for r in rows),
                 max_gradient_envelope_violation=max(r['actual_error']-r['eta'] for r in rows),
                 max_robust_gap_violation=max(r['actual_gap']-r['robust_bound'] for r in rows),
                 derivative_naive_underbounds=sum(r['panel']=='derivative' and r['naive_bound']<r['actual_gap']-1e-12 for r in rows),
                 min_nonapex_polar_norm=min(r['qp']['polar_norm'] for r in rows if r['qp']['mode']=='SLSQP_nonapex'),
                 min_nonapex_solution_norm=min(r['qp']['solution_norm'] for r in rows if r['qp']['mode']=='SLSQP_nonapex'),
                 min_boundary_normal_polar_norm=min(r['qp']['polar_norm'] for r in rows if r['qp'].get('normal_applied',False)),
                 retained_cases=len(rows),nonapex_cases=sum(r['qp']['mode']=='SLSQP_nonapex' for r in rows),
                 solver_status_counts={str(s):sum(r['qp']['status']==s for r in rows) for s in sorted({r['qp']['status'] for r in rows})})
    report=dict(status='PASS' if all(tests.values()) and all(auxiliary_tests.values()) else 'FAIL',scope='Controlled sensitivity of a fixed quadratic per metric; no trained-network performance',
                assumptions='Supplied epsb,epsv and analytic prior G; analytic HVP is perturbed by deterministic errors within these envelopes. FD HVP is an independent formula check, not the source of certified envelopes.',
                derivative_noise='delta_b=-0.75 epsb, delta_v=0.8 epsv e',
                wrapper='Constructed derivative-only interface(raw=(I,w))->value,rawgrad,v,b,W. Physical g is exposed only by private validation oracle. Known quadratic and supplied analytic G also make direct g available in principle; this is not evidence of practical API necessity.',
                KKT_definition='For a nonapex boundary optimum p, h=pI-||pp||, n=(-1,pp/||pp||), lambda=max(0,-g(p).n/(n.n)); residual=||g(p)+lambda*n||_infinity; complementarity=abs(lambda*h). Interior points use lambda=0. Apex is certified separately iff g(0) belongs to the self-dual Lorentz cone; no single-normal fallback is used for apex KKT.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                software=dict(numpy=np.__version__,scipy=scipy.__version__),FD_step=2e-5,
                tests=tests,summary=summary,records=rows,apex_checks=apexchecks,
                separate_pointwise_tests=auxiliary_tests,separate_pointwise_records=auxiliary,
                pointwise_formulae='f_tau(u)=f(u)+tau*n^T(u-s); at exact fold scalar cr=e^T Hww e gives gp=(gw-cr*e)/I and gI=b+cr/I. Physical Hessian remains W.')
    (HERE/'results/robust_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],tests=tests,summary=summary,separate_pointwise_tests=auxiliary_tests),indent=2))
    assert all(tests.values()) and all(auxiliary_tests.values()),'Check retained records; no figure is produced after failed verification'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(7.0,5.0),layout='constrained')
    for sheet,color in [('inner','#0072B2'),('outer','#D55E00')]:
        rr=[r for r in rows if r['panel']=='radius' and r['sheet']==sheet]
        axs[0,0].loglog([r['x'] for r in rr],[r['actual_error'] for r in rr],color=color,label=sheet+' error')
        axs[0,0].loglog([r['x'] for r in rr],[r['eta'] for r in rr],'--',color=color,label=sheet+' envelope')
    axs[0,0].set(xlabel=r'Radial distance $|r-1|$',ylabel=r'Gradient error in $W^{-1}$ norm',title='(a) Near-fold geometry (exact derivatives)')
    rr=[r for r in rows if r['panel']=='derivative']
    axs[0,1].semilogx([r['x'] for r in rr],[r['naive_bound']/r['actual_gap'] for r in rr],color='#D55E00',label='Uncorrected bound / gap')
    axs[0,1].semilogx([r['x'] for r in rr],[r['robust_bound']/r['actual_gap'] for r in rr],color='#0072B2',label='Robust bound / gap')
    axs[0,1].axhline(1.,color='.3',ls=':',lw=.8)
    axs[0,1].set(xlabel=r'Supplied envelopes $\epsilon_b=\epsilon_v$',ylabel='Bound / Independently computed gap',title='(b) Controlled derivative perturbations')
    for ax,panel,title,xlabel in [(axs[1,0],'intensity','(c) Small-intensity amplification',r'Intensity $I$'),(axs[1,1],'metric','(d) Metric conditioning',r'Condition number $\kappa(W)$')]:
        rr=[r for r in rows if r['panel']==panel]
        ax.loglog([r['x'] for r in rr],[r['actual_error'] for r in rr],color='#D55E00',label='Actual gradient error')
        ax.loglog([r['x'] for r in rr],[r['eta'] for r in rr],'--',color='#0072B2',label='Supplied-error envelope')
        ax.set(xlabel=xlabel,ylabel=r'Gradient error in $W^{-1}$ norm',title=title)
    for ax in axs.flat:ax.grid(alpha=.18);ax.legend(fontsize=6.4)
    fig.savefig(HERE.parent/'latex/images/sensitivity.pdf');fig.savefig(HERE.parent/'latex/images/sensitivity.png',dpi=230)
    plt.close(fig)


if __name__=='__main__':main()
