"""Explicit finite-tolerance bounds for an inner rational Stokes chart.

These evaluate real-arithmetic formulas in float64. Supplied derivative error
budgets, curvature/convexity bounds and geometry are assumptions, not inferred
rounding-error certificates. No optimal point or raw observations are needed.
"""
import math
import numpy as np


def geometry(r):
    d=1+r*r
    return 2*r/d,2*(1-r*r)/(d*d),2/d,4*r*(r*r-3)/(d*d*d)


def tolerances(I,b_hat,j_hat,c_hat=None,error_b=0.,error_j=0.,error_c=0.):
    assert I>0 and min(error_b,error_j,error_c)>=0
    eps=max(abs(b_hat)+error_b,(float(np.linalg.norm(j_hat))+error_j)/I)
    nu=max(0.,(-c_hat+error_c)/I) if c_hat is not None else None
    return eps,nu


def stopping_bound(I,r,mu,L,eps,nu=None,Ibar=None):
    assert I>0 and 0<=r<=1 and mu>0 and L>=mu and eps>=0
    Ibar=I if Ibar is None else Ibar
    assert I<=Ibar
    phi,a,t,kappa=geometry(r)
    first=((eps+phi*eps/a)**2+(eps/a)**2+(eps/t)**2)/(2*mu) if r<1 else math.inf
    near=math.inf
    if r>=.5:
        assert nu is not None and nu>=0
        mplus=(nu+I*L*a*a)/abs(kappa)
        D=(1-r)*(1+r*r)/(2*(1+r))
        positive=((eps+phi*mplus)**2+mplus*mplus+(eps/t)**2)/(2*mu)
        negative=((eps+eps*D)**2+(eps/t)**2)/(2*mu)+I*eps*D
        near=max(positive,negative)
    B=Ibar*L
    uniform=52*B**(2/3)*eps**(4/3)/mu+3*(nu or 0.)**2/mu if eps<=B/8 and (r<.5 or nu is not None) else math.inf
    return dict(first_order=first,one_sided_curvature=near,pointwise=min(first,near),
                uniform=uniform,best=min(first,near,uniform))


def target_thresholds(mu,L,Ibar,gap):
    assert min(mu,L,Ibar,gap)>0 and L>=mu
    B=Ibar*L
    return dict(eps=min(B/8,(mu*gap/(104*B**(2/3)))**.75),nu=math.sqrt(mu*gap/6))
