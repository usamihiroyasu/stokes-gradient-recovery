#!/usr/bin/env python3
"""Independent exact-rational checks for the fixed-point interval arithmetic.

The exponential reference deliberately uses a Taylor series at the original
argument, without range reduction or repeated squaring.  Fraction arithmetic
and a geometric remainder bound make this an enclosure oracle, not a decimal
approximation.  This file does not inspect scientific input/output records.
"""
from __future__ import annotations

from fractions import Fraction
from math import isqrt
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Any, Callable


def ceil_fraction(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def exp_fraction_enclosure(
    x: Fraction, bits: int
) -> tuple[Fraction, Fraction, int]:
    """Enclose exp(x), using exact arithmetic and an absolute tail target."""
    if x == 0:
        return Fraction(1), Fraction(1), 0
    if x < 0:
        lo, hi, count = exp_fraction_enclosure(-x, bits)
        return 1 / hi, 1 / lo, count
    target = Fraction(1, 1 << bits)
    term = Fraction(1)
    total = Fraction(1)
    n = 0
    while True:
        next_term = term * x / (n + 1)
        ratio_bound = x / (n + 2)
        if ratio_bound < 1:
            remainder = next_term / (1 - ratio_bound)
            if remainder <= target:
                return total, total + remainder, n
        n += 1
        term = next_term
        total += term
        if n > 4096:
            raise AssertionError("exponential oracle did not reach its tail target")


def fraction_extrema(
    op: str, a: tuple[int, int], b: tuple[int, int], q: int
) -> tuple[Fraction, Fraction]:
    aa = [Fraction(a[0], q), Fraction(a[1], q)]
    bb = [Fraction(b[0], q), Fraction(b[1], q)]
    if op == "add":
        values = [x + y for x in aa for y in bb]
    elif op == "sub":
        values = [x - y for x in aa for y in bb]
    elif op == "mul":
        values = [x * y for x in aa for y in bb]
    elif op == "div":
        if b[0] <= 0 <= b[1]:
            raise ZeroDivisionError("the reference denominator contains zero")
        values = [x / y for x in aa for y in bb]
    else:
        raise ValueError(op)
    return min(values), max(values)


def tight_scaled_bounds(lo: Fraction, hi: Fraction, q: int) -> tuple[int, int]:
    return (lo * q).numerator // (lo * q).denominator, ceil_fraction(hi * q)


def sqrt_integer_bounds(a: tuple[int, int], q: int) -> tuple[int, int]:
    if a[0] < 0:
        raise ValueError("square root lower endpoint is negative")
    low = isqrt(a[0] * q)
    high = isqrt(a[1] * q)
    if high * high != a[1] * q:
        high += 1
    return low, high


def run_arithmetic_checks(
    *,
    precision: int,
    make: Callable[[int, int], Any],
    bounds: Callable[[Any], tuple[int, int]],
    sqrt: Callable[[Any], Any],
    exp: Callable[[Any], Any],
) -> dict[str, Any]:
    """Check an implementation through a small adapter, without its internals."""
    q = 1 << precision
    counts = {key: 0 for key in ("add", "sub", "mul", "div", "neg", "square", "abs", "sqrt", "exp", "rejected")}
    rng = random.Random(20261022 + precision)

    def equal(label: str, value: Any, expected: tuple[int, int]) -> None:
        actual = bounds(value)
        if actual != expected:
            raise AssertionError(f"{label}: scaled endpoints {actual} != exact {expected}")
        counts[label.split(":", 1)[0]] += 1

    def rejected(label: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except (ValueError, ZeroDivisionError):
            counts["rejected"] += 1
        else:
            raise AssertionError(f"{label}: invalid real-domain operation was accepted")

    rejected("reversed interval", lambda: make(1, 0))
    special = [
        (-2*q, -q), (-q, -q), (-q, q), (-1, 0), (-1, -1),
        (0, 0), (0, 1), (1, 1), (q-1, q+1), (q, q),
        (q, 2*q), (-2*q, -1), (1, 2*q), (-2*q-1, -q+1),
        (2*q-1, 2*q+1),
    ]
    pairs = [(a, b) for a in special for b in special]
    for _ in range(300):
        a = tuple(sorted((rng.randrange(-4*q, 4*q+1), rng.randrange(-4*q, 4*q+1))))
        b = tuple(sorted((rng.randrange(-4*q, 4*q+1), rng.randrange(-4*q, 4*q+1))))
        pairs.append((a, b))
    for a, b in pairs:
        ai, bi = make(*a), make(*b)
        for op, call in (
            ("add", lambda: ai + bi), ("sub", lambda: ai - bi),
            ("mul", lambda: ai * bi), ("div", lambda: ai / bi),
        ):
            if op == "div" and b[0] <= 0 <= b[1]:
                rejected(f"div:{a}/{b}", call)
                continue
            exact = fraction_extrema(op, a, b, q)
            equal(f"{op}:{a},{b}", call(), tight_scaled_bounds(*exact, q))
        equal(f"neg:{a}", -ai, (-a[1], -a[0]))
        square_values = [Fraction(a[0], q)**2, Fraction(a[1], q)**2]
        if a[0] <= 0 <= a[1]:
            square_values.append(Fraction(0))
        equal(f"square:{a}", ai.square(), tight_scaled_bounds(min(square_values), max(square_values), q))
        abs_values = [abs(a[0]), abs(a[1])]
        if a[0] <= 0 <= a[1]:
            abs_values.append(0)
        equal(f"abs:{a}", abs(ai), (min(abs_values), max(abs_values)))

    square_inputs = [(0, 0), (0, 1), (1, 1), (q-1, q+1), (q, q), (4*q, 4*q), (q+1, 4*q-1)]
    square_inputs += [tuple(sorted((rng.randrange(16*q), rng.randrange(16*q)))) for _ in range(200)]
    for a in square_inputs:
        result = sqrt(make(*a))
        equal(f"sqrt:{a}", result, sqrt_integer_bounds(a, q))
        low, high = bounds(result)
        # These exact inequalities also verify enclosure directly, separately
        # from the isqrt construction used to specify the tight endpoints.
        assert low >= 0 and low * low <= a[0] * q
        assert high * high >= a[1] * q
    for a in [(-1, 0), (-q, q), (-q, -1)]:
        rejected(f"sqrt:{a}", lambda a=a: sqrt(make(*a)))

    # Include exact reduction thresholds, their adjacent grid points, the domain
    # endpoints, and the smallest representable positive/negative arguments.
    points = {0, 1, -1, -64*q, 64*q}
    for center in (q//16, q//8, q//4, q//2, q, 2*q, 32*q):
        for offset in (-1, 0, 1):
            points.add(center + offset)
            points.add(-center - offset)
    exponential_inputs = [(x, x) for x in sorted(points)]
    exponential_inputs += [(-64*q, 64*q), (-q, q), (-1, 1), (q//8-1, q//8+1)]
    # A few nonspecial dyadic endpoint intervals exercise the monotone wrapper.
    exponential_inputs += [tuple(sorted((rng.randrange(-64*q, 64*q+1), rng.randrange(-64*q, 64*q+1)))) for _ in range(6)]
    oracle_cache: dict[int, tuple[Fraction, Fraction, int]] = {}
    max_taylor_degree = 0
    for a in exponential_inputs:
        got = bounds(exp(make(*a)))
        for endpoint in a:
            if endpoint not in oracle_cache:
                oracle_cache[endpoint] = exp_fraction_enclosure(Fraction(endpoint, q), precision + 128)
            max_taylor_degree = max(max_taylor_degree, oracle_cache[endpoint][2])
        lower_oracle = oracle_cache[a[0]][0]
        upper_oracle = oracle_cache[a[1]][1]
        if got[0] < 0 or got[1] <= 0 or got[0] > got[1]:
            raise AssertionError(f"exp:{a}: invalid positive enclosure {got}")
        if Fraction(got[0], q) > lower_oracle or Fraction(got[1], q) < upper_oracle:
            raise AssertionError(f"exp:{a}: does not contain the independent exact-rational enclosure")
        counts["exp"] += 1
    for a in [(-64*q-1, -64*q), (64*q, 64*q+1), (-65*q, 65*q)]:
        rejected(f"exp:{a}", lambda a=a: exp(make(*a)))
    return {
        "precision_bits": precision,
        "counts": counts,
        "exp_oracle": "exact Fraction Taylor at original argument; geometric tail; no range reduction",
        "exp_oracle_tail_bits": precision + 128,
        "exp_oracle_max_degree": max_taylor_degree,
        "passed": True,
    }


def api_edge_checks(module: Any, ctx: Any) -> int:
    """Verify scalar reverse operations, powers, display bounds, and rejection."""
    q, count = ctx.q, 0
    for a, b in [(-2, -1), (-2, 3), (-1, 1), (0, 1), (1, 2)]:
        x = ctx.interval(a, b)
        for n in range(9):
            values = [Fraction(a)**n, Fraction(b)**n]
            if n and n % 2 == 0 and a <= 0 <= b:
                values.append(Fraction(0))
            z = x**n
            assert Fraction(z.lo, q) <= min(values) and Fraction(z.hi, q) >= max(values)
            count += 1
        for z, lo, hi in [(2+x, 2+a, 2+b), (2-x, 2-b, 2-a), (2*x, 2*a, 2*b)]:
            assert z.contains_fraction(lo) and z.contains_fraction(hi)
            count += 1
        if not a <= 0 <= b:
            z = 2 / x
            values = [Fraction(2, a), Fraction(2, b)]
            assert z.contains_fraction(min(values)) and z.contains_fraction(max(values))
            count += 1
    for a, b in [(Fraction(1, 3), Fraction(2, 3)), (Fraction(-1, 3), Fraction(1, 3)), (Fraction(-3, 2), Fraction(-1, 2))]:
        x = ctx.interval(a, b)
        assert Fraction(x.lower_float()) <= Fraction(x.lo, q)
        assert Fraction(x.upper_float()) >= Fraction(x.hi, q)
        count += 1
    for call in [
        lambda: ctx.interval(float("inf")), lambda: ctx.interval(float("nan")),
        lambda: ctx.interval(1) + module.Context(ctx.bits+1).interval(1),
        lambda: ctx.interval(1)**-1, lambda: ctx.interval(1)**Fraction(1, 2),
    ]:
        try:
            call()
        except (ValueError, ZeroDivisionError):
            count += 1
        else:
            raise AssertionError("expected API edge rejection")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    arithmetic_path = root / "theory" / "validated_stopping" / "arithmetic.py"
    spec = importlib.util.spec_from_file_location("independent_interval_target", arithmetic_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the arithmetic module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    reports = []
    for p in args.bits:
        ctx = module.Context(bits=p)
        assert ctx.q == 1 << p
        q = ctx.q
        make = lambda lo, hi, ctx=ctx, q=q: ctx.interval(Fraction(lo, q), Fraction(hi, q))
        report = run_arithmetic_checks(
            precision=p,
            make=make,
            bounds=lambda obj: (obj.lo, obj.hi),
            sqrt=lambda obj: obj.sqrt(),
            exp=lambda obj: obj.exp(),
        )
        # Scalar input conversion is checked separately from arithmetic.
        for value in [Fraction(1, 3), Fraction(-1, 3), Fraction(1, 10), Fraction(0), Fraction(64), Fraction(1, 1 << (p+2))]:
            interval = ctx.interval(value)
            expected = tight_scaled_bounds(value, value, q)
            assert (interval.lo, interval.hi) == expected
            assert interval.contains_fraction(value)
        for value in [0.1, -0.1, 0.0, 1.0]:
            exact = Fraction(value)
            interval = ctx.interval(value)
            assert (interval.lo, interval.hi) == tight_scaled_bounds(exact, exact, q)
            assert interval.contains_fraction(exact)
        report["scalar_conversion_checks"] = 10
        report["api_edge_checks"] = api_edge_checks(module, ctx)
        reports.append(report)
    result = {
        "passed": True,
        "scope": "arithmetic enclosure checks only; no scientific inputs or proof review",
        "arithmetic_sha256": hashlib.sha256(arithmetic_path.read_bytes()).hexdigest(),
        "independent_checks_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "reports": reports,
    }
    output = args.output or Path(__file__).with_name("INDEPENDENT_ARITHMETIC_CHECKS.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
