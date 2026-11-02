"""Mixed scalar recovery, supplied-input errors, and restricted-query ambiguity."""
from pathlib import Path
from decimal import Decimal as D, localcontext
import hashlib,json
from corrected_recovery_checks import dot,loss_grad,head_values

HERE=Path(__file__).resolve().parent

def mixed(I,e,r,b,j,c,h):
    phi=2*r/(1+r*r); a=2*(1-r*r)/(1+r*r)**2; t=2/(1+r*r)
    k=4*r*(r*r-3)/(1+r*r)**3
    m=(c/I-I*a*a*h)/k; je=dot(e,j)
    gp=[(j[i]-e[i]*je)/(I*t)+e[i]*m for i in range(2)]
    return [b-phi*m]+gp

def main():
    recovery=[];errors=[];ambiguity=[]
    with localcontext() as ctx:
        ctx.prec=80
        e=[D('.6'),D('.8')]; ep=[-e[1],e[0]]
        h1=D('1e-18');h2=D('1e-12')
        for kind in ['quartic','exponential_quartic','softplus_coupled']:
            for I in [D('.2'),D('1.2')]:
                for r in [D('.8'),D('.99'),D(1),D('1.01'),D('1.2')]:
                    w=[r*x for x in e]; physical=lambda s:loss_grad(s,kind)
                    f,b,j,g=head_values(I,w,physical)
                    s=[I]+[I*2*x/(1+r*r) for x in w]
                    sm=[s[0]]+[s[i+1]-h1*e[i] for i in range(2)]
                    sp=[s[0]]+[s[i+1]+h1*e[i] for i in range(2)]
                    gm=physical(sm)[1][1:];gp=physical(sp)[1][1:]
                    hrr=dot(e,[(gp[i]-gm[i])/(2*h1) for i in range(2)])
                    fm=head_values(I,[(r-h2)*x for x in e],physical)[0]
                    fp=head_values(I,[(r+h2)*x for x in e],physical)[0]
                    cr=(fp-2*f+fm)/(h2*h2)
                    base=mixed(I,e,r,b,j,cr,hrr)
                    err=max(abs(x-y) for x,y in zip(base,g));assert err<D('1e-18')
                    recovery.append(dict(objective=kind,I=float(I),r=float(r),h_rr=float(hrr),c_r=float(cr),
                        physical_gradient=[float(x) for x in g],recovered=[float(x) for x in base],error=float(err)))
                    eps=D('1e-6');epsh=D('1e-3')
                    db=-D('.75')*eps;dj=[D('.8')*eps*x for x in ep];dc=D('.6')*eps;dh=-D('.5')*epsh
                    shifted=mixed(I,e,r,b+db,[j[i]+dj[i] for i in range(2)],cr+dc,hrr+dh)
                    delta=[shifted[i]-base[i] for i in range(3)]
                    phi=2*r/(1+r*r);a=2*(1-r*r)/(1+r*r)**2;t=2/(1+r*r);k=4*r*(r*r-3)/(1+r*r)**3
                    beta=-I*a*a/k;dm=dc/(I*k)+beta*dh
                    predicted=[db-phi*dm]+[dj[i]/(I*t)+e[i]*dm for i in range(2)]
                    residual=max(abs(x-y) for x,y in zip(delta,predicted));assert residual<D('1e-60')
                    envelope=eps+eps/(I*t)+(1+phi*phi).sqrt()*(eps/(I*abs(k))+abs(beta)*epsh)
                    actual=dot([shifted[i]-g[i] for i in range(3)],[shifted[i]-g[i] for i in range(3)]).sqrt()
                    assert actual<=envelope+D('1e-18')
                    errors.append(dict(objective=kind,I=float(I),r=float(r),identity_residual=float(residual),
                        error=float(actual),envelope=float(envelope),beta_parallel=float(beta)))
        # Centered affine/quadratic perturbations preserve b, Pj and c, but alter g.
        I=D('1.2');g=[D('.4'),D('.2'),D('-.3')];H=[[D('.5'),D('.1')],[D('.1'),D('.8')]]
        for r in [D('.8'),D('.99'),D('1.01'),D('1.2')]:
            phi=2*r/(1+r*r);a=2*(1-r*r)/(1+r*r)**2;t=2/(1+r*r);k=4*r*(r*r-3)/(1+r*r)**3
            hrr=dot(e,[dot(row,e) for row in H])
            def observations(gg,hh):
                m=dot(e,gg[1:]);b=gg[0]+phi*m
                j=[I*(a*e[i]*m+t*(gg[i+1]-e[i]*m)) for i in range(2)]
                tangent=[j[i]-e[i]*dot(e,j) for i in range(2)]
                c=I*I*a*a*hh+I*k*m
                return [b]+tangent+[c],j
            obs,j=observations(g,hrr)
            for tau in [D('-.1'),D('.1')]:
                dg=[-phi*tau]+[tau*x for x in e];dh=-k*tau/(I*a*a)
                gg=[g[i]+dg[i] for i in range(3)];changed,jj=observations(gg,hrr+dh)
                residual=max(abs(x-y) for x,y in zip(obs,changed));assert residual<D('1e-60')
                first_difference=[jj[i]-j[i] for i in range(2)]
                assert max(abs(first_difference[i]-I*a*tau*e[i]) for i in range(2))<D('1e-60')
                ambiguity.append(dict(r=float(r),tau=float(tau),gradient_delta=[float(x) for x in dg],
                    h_rr_delta=float(dh),same_observation_residual=float(residual),full_first_gradient_changes=True))
    report=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Known exact geometry; mixed observation tuple. Decimal differences are consistency checks.',
        precision=80,recovery_records=recovery,error_records=errors,ambiguity_records=ambiguity,
        summary=dict(recovery_cases=len(recovery),error_cases=len(errors),ambiguity_cases=len(ambiguity),
            max_recovery_error=max(x['error'] for x in recovery),max_error_identity_residual=max(x['identity_residual'] for x in errors),
            max_ambiguity_observation_residual=max(x['same_observation_residual'] for x in ambiguity)))
    (HERE/'results/mixed_recovery_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
