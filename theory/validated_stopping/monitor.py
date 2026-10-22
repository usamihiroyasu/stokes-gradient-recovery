"""Validated nominal four-analyzer stopping at exact binary input values.

The model is the paper's ideal unweighted quadratic with mu=L=1/2.
All computations determining acceptance use dyadic intervals. Converting
the original outer state to the exact inner preimage preserves the physical
state algebraically; no rounded optimizer restart is performed here.
"""
from fractions import Fraction as F
from math import isfinite
from arithmetic import Context,Interval,maximum,minimum


def _input(x,y):
    if len(x)!=3 or len(y)!=4 or not all(isfinite(v) for v in [*x,*y]):
        raise ValueError('three finite raw coordinates and four finite observations required')
    return [F(v) for v in x],[F(v) for v in y]


def evaluate(x,y,bits=128,target=F(1,1000000)):
    """Return a certified upper bound or an explicit inconclusive result.

    Certified means the *mathematical nominal objective* at the state mapped
    from the exact supplied binary raw coordinates has gap <= exact target.
    It does not certify unknown instrument parameters or measured truth.
    """
    try:
        xx,yy=_input(x,y);ctx=Context(bits);C=ctx.interval
        I=C(xx[0]).exp()
        if I.lo<=0 or I.hi>8*ctx.q:
            raise ValueError('intensity enclosure outside 0 < I <= 8')
        w=[C(v) for v in xx[1:]];r2=sum(v*v for v in xx[1:])
        original_squared=C(r2);outer=r2>1
        s=[I,*[2*I*v/(1+original_squared) for v in w]]
        z=[C(sum(yy)/2),C(yy[0]-yy[2]),C(yy[1]-yy[3])]
        g=[s[0]-z[0],(s[1]-z[1])/2,(s[2]-z[2])/2]
        if r2==0:
            # Exact original coordinate w=0: DF=2 Id, and b=g_I.
            r=C(0);phi=C(0);ar=C(2);tr=C(2)
            eps=maximum(abs(g[0]),2*(g[1].square()+g[2].square()).sqrt())
            first=((eps+phi*eps/ar).square()+(eps/ar).square()+(eps/tr).square())
            bound=first;nu=None;curvature=False
        else:
            rr=original_squared.sqrt()
            e=[v/rr for v in w]
            r=1/rr if outer else rr
            # Proven domain intersection: min(sqrt(r2),1/sqrt(r2)) <= 1.
            r=Interval(max(0,r.lo),min(ctx.q,r.hi),ctx)
            den=1+r.square();phi=2*r/den
            ar=2*(1-r.square())/den.square();tr=2/den
            m=e[0]*g[1]+e[1]*g[2]
            u=-e[1]*g[1]+e[0]*g[2]
            b=g[0]+phi*m
            eps=maximum(abs(b),((ar*m).square()+(tr*u).square()).sqrt())
            inner_squared=1/r2 if outer else r2
            first=None
            if inner_squared<1 and ar.lo>0:
                first=(eps+phi*eps/ar).square()+(eps/ar).square()+(eps/tr).square()
            curvature=inner_squared>=F(1,4);nu=None;near=None
            if curvature:
                k=4*r*(3-r.square())/(den**3)
                # Scalar c_r / I only; no full head Hessian is formed.
                curvature_over_I=I*ar.square()/2-k*m
                nu=(-curvature_over_I).positive()
                M=(nu+I*ar.square()/2)/k
                D=(1-r)*den/(2*(1+r))
                # Proven inner-domain nonnegativity; dependency inflation may straddle zero.
                D=Interval(max(0,D.lo),D.hi,ctx)
                plus=(eps+phi*M).square()+M.square()+(eps/tr).square()
                minus=(eps+eps*D).square()+(eps/tr).square()+I*eps*D
                near=maximum(plus,minus)
            candidates=[v for v in [first,near] if v is not None]
            if not candidates:raise ValueError('insufficiently resolved chart denominators')
            bound=minimum(*candidates)
        # 2*mu=1 in every displayed pointwise formula above.
        certified=bound.hi*target.denominator<=target.numerator*ctx.q
        return dict(certified=certified,status='certified' if certified else 'bound_above_target',
            bound_upper=bound.upper_float(),bound_upper_integer=str(bound.hi),bits=bits,
            epsilon_upper=eps.upper_float(),nu_upper=nu.upper_float() if nu is not None else None,
            scalar_curvature_evaluations=int(curvature),full_head_hessians=0,
            exact_inner_preimage_used=outer,intensity_upper=I.upper_float(),radius_upper=r.upper_float())
    except (ValueError,ZeroDivisionError,ArithmeticError,OverflowError) as error:
        return dict(certified=False,status='inconclusive',reason=str(error),bits=bits,
                    full_head_hessians=0)


def reference_gap(x,y,bits=256):
    """Independent direct-cone formula in outward intervals, for auditing only."""
    xx,yy=_input(x,y);ctx=Context(bits);C=ctx.interval
    I=C(xx[0]).exp();r2=sum(v*v for v in xx[1:])
    s=[I,*[2*I*C(v)/(1+C(r2)) for v in xx[1:]]]
    zi=sum(yy)/2;zq=yy[0]-yy[2];zu=yy[1]-yy[3];rho2=zq*zq+zu*zu
    val=((s[0]-C(zi)).square()+((s[1]-C(zq)).square()+(s[2]-C(zu)).square())/2)/2
    if zi>=0 and zi*zi>=rho2:
        optimum=C(0)
    elif zi<=0 and 4*zi*zi>=rho2:
        optimum=C(zi*zi/2+rho2/4)
    else:
        optimum=(C(rho2).sqrt()-C(zi)).square()/6
    gap=(val-optimum).positive()
    return gap
