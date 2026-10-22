"""Exact algebra and limit checks for the proof, using SymPy rationals."""
from pathlib import Path
import json
import sympy as s
P=Path(__file__).resolve().parent
r=s.symbols('r',positive=True);x=s.symbols('x',nonnegative=True)
phi=2*r/(1+r*r);a=s.diff(phi,r);kap=s.diff(a,r);delta=1-r
identities={
 'first_inverse_lower':s.factor(a-delta),
 'near_first_inverse_upper':s.factor(2*delta-a),
 'radial_curvature_lower':s.factor(-kap-1),
 'slack_ratio':s.factor((1-phi)/a),
}
assert s.simplify((a-delta)-delta**2*(r**3+r*r+3*r+1)/(1+r*r)**2)==0
assert s.simplify((2*delta-a)-2*delta*r*(r**3+2*r-1)/(1+r*r)**2)==0
assert s.simplify((-kap-1)-delta*(r**5+r**4+4*r**3+8*r*r+11*r-1)/(1+r*r)**3)==0
for poly in [r**3+2*r-1,r**5+r**4+4*r**3+8*r*r+11*r-1]:
    assert all(c>=0 for c in s.Poly(s.expand(poly.subs(r,x+s.Rational(1,2))),x).all_coeffs())
I=(1+phi)/(1+phi*phi);m=s.factor(I*phi-1)
gap=s.factor(((I-1)**2+m*m)/2);eps=s.factor(-a*m)
assert s.simplify(gap-delta**4/(2*(r**4+6*r*r+1)))==0
assert s.limit(eps/delta**3,r,1,dir='-')==s.Rational(1,4)
assert s.limit(gap/delta**4,r,1,dir='-')==s.Rational(1,16)
Io=(1+phi/2)/(1+phi*phi);mo=Io*phi-s.Rational(1,2)
outer_eps=a*mo;outer_nu=-Io*a*a-kap*mo
outer_gap=((Io-1)**2+mo*mo)/2
assert s.limit(outer_eps*r*r,r,s.oo)==1
assert s.limit(outer_nu*r**3,r,s.oo)==2
assert s.limit(outer_gap,r,s.oo)==s.Rational(1,8)
assert s.Rational(5,2)+48+1<s.Integer(52)
out=dict(sympy=s.__version__,exact_rational_identities_verified=True,
         shifted_polynomial_coefficients_nonnegative=True,
         gradient_limit='1/4',gap_limit='1/16',sharp_ratio='2^(-4/3)',
         outer_epsilon_limit='1',outer_nu_limit='2',outer_gap_limit='1/8',
         conservative_uniform_coefficient='52 > 103/2',
         factorizations={k:str(v) for k,v in identities.items()},
         scope='Exact identity/limit verification supplements the written proof; not formal theorem-prover certification or novelty verification.')
(P/'results/SYMBOLIC_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
