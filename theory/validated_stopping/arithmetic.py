"""Outward dyadic intervals backed only by exact Python integers.

Endpoints are integer multiples of 2**(-bits). No hardware rounding mode,
libm function, Decimal transcendental, or floating-point arithmetic is used
in interval operations. Float inputs denote their exact binary values.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, isqrt, nextafter, inf


def ceildiv(n, d):
    return -((-n)//d)


@dataclass(frozen=True)
class Context:
    bits: int = 128

    def __post_init__(self):
        if not isinstance(self.bits, int) or self.bits < 8:
            raise ValueError('bits must be an integer >= 8')

    @property
    def q(self):
        return 1 << self.bits

    def interval(self, lo, hi=None):
        if isinstance(lo, Interval):
            if lo.ctx != self or hi is not None:
                raise ValueError('incompatible interval context')
            return lo
        if isinstance(lo, float) and not isfinite(lo):
            raise ValueError('finite inputs required')
        if isinstance(hi, float) and not isfinite(hi):
            raise ValueError('finite inputs required')
        a=Fraction(lo); b=a if hi is None else Fraction(hi)
        if a>b:
            raise ValueError('reversed endpoints')
        return Interval((a.numerator*self.q)//a.denominator,
                        ceildiv(b.numerator*self.q,b.denominator),self)


@dataclass(frozen=True)
class Interval:
    lo: int
    hi: int
    ctx: Context

    def __post_init__(self):
        if self.lo>self.hi:
            raise ValueError('reversed endpoints')

    def _other(self, other):
        return self.ctx.interval(other)

    def __add__(self, other):
        o=self._other(other)
        return Interval(self.lo+o.lo,self.hi+o.hi,self.ctx)

    __radd__=__add__

    def __neg__(self):
        return Interval(-self.hi,-self.lo,self.ctx)

    def __sub__(self, other):
        return self+-self._other(other)

    def __rsub__(self, other):
        return self._other(other)+-self

    def __mul__(self, other):
        o=self._other(other); q=self.ctx.q
        products=[a*b for a in (self.lo,self.hi) for b in (o.lo,o.hi)]
        return Interval(min(products)//q,ceildiv(max(products),q),self.ctx)

    __rmul__=__mul__

    def __truediv__(self, other):
        o=self._other(other); q=self.ctx.q
        if o.lo<=0<=o.hi:
            raise ZeroDivisionError('denominator interval contains zero')
        pairs=[(a*q,b) for a in (self.lo,self.hi) for b in (o.lo,o.hi)]
        return Interval(min(n//d for n,d in pairs),max(ceildiv(n,d) for n,d in pairs),self.ctx)

    def __rtruediv__(self, other):
        return self._other(other)/self

    def square(self):
        vals=[self.lo*self.lo,self.hi*self.hi];q=self.ctx.q
        lower=0 if self.lo<=0<=self.hi else min(vals)//q
        return Interval(lower,ceildiv(max(vals),q),self.ctx)

    def __pow__(self, n):
        if not isinstance(n,int) or n<0:
            raise ValueError('nonnegative integer powers only')
        result=self.ctx.interval(1);base=self
        while n:
            if n&1:result=result*base
            n//=2
            if n:base=base.square()
        return result

    def __abs__(self):
        return Interval(0 if self.lo<=0<=self.hi else min(abs(self.lo),abs(self.hi)),
                        max(abs(self.lo),abs(self.hi)),self.ctx)

    def sqrt(self):
        if self.lo<0:
            raise ValueError('square-root interval has negative lower endpoint')
        q=self.ctx.q;lo=isqrt(self.lo*q);hi=isqrt(self.hi*q)
        if hi*hi<self.hi*q:hi+=1
        return Interval(lo,hi,self.ctx)

    def exp(self):
        # Monotonicity permits separate endpoint evaluation, including a sign change.
        if self.lo < -64*self.ctx.q or self.hi > 64*self.ctx.q:
            raise ValueError('supported exponential argument domain is [-64,64]')
        lower=_exp_endpoint(self.lo,self.ctx).lo
        upper=_exp_endpoint(self.hi,self.ctx).hi
        return Interval(lower,upper,self.ctx)

    def positive(self):
        return Interval(max(0,self.lo),max(0,self.hi),self.ctx)

    def upper_float(self):
        f=float(Fraction(self.hi,self.ctx.q))
        if isfinite(f) and Fraction(f)<Fraction(self.hi,self.ctx.q):
            f=nextafter(f,inf)
        return f

    def lower_float(self):
        f=float(Fraction(self.lo,self.ctx.q))
        if isfinite(f) and Fraction(f)>Fraction(self.lo,self.ctx.q):
            f=nextafter(f,-inf)
        return f

    def contains_fraction(self, x):
        v=Fraction(x)*self.ctx.q
        return self.lo<=v<=self.hi


def _exp_endpoint(n,ctx):
    if n==0:return ctx.interval(1)
    if n<0:return 1/_exp_endpoint(-n,ctx)
    # Choose k by an exact integer comparison; t <= 1/8.
    k=0
    while 8*n>ctx.q*(1<<k):k+=1
    t=Interval(n,n,ctx)/(1<<k)
    if t.hi*8>ctx.q:raise ArithmeticError('range-reduction enclosure too wide')
    term=ctx.interval(1);total=term
    for j in range(1,49):
        term=term*t/j;total=total+term
    next_term=term*t/49
    tail=next_term/(1-t/50)
    total=Interval(total.lo,total.hi+tail.hi,ctx)
    for _ in range(k):total=total.square()
    return total


def maximum(*args):
    ctx=args[0].ctx
    if any(x.ctx!=ctx for x in args):raise ValueError('incompatible contexts')
    return Interval(max(x.lo for x in args),max(x.hi for x in args),ctx)


def minimum(*args):
    ctx=args[0].ctx
    if any(x.ctx!=ctx for x in args):raise ValueError('incompatible contexts')
    return Interval(min(x.lo for x in args),min(x.hi for x in args),ctx)
