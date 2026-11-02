# Outward evaluation of the Stokes stopping bound

This standard-library implementation evaluates the paper's pointwise bound
with outward dyadic intervals. It supports the nominal ideal four-analyzer
least-squares model, with metric `diag(1, 1/2, 1/2)` and `I = exp(a)`.
Supplement S21 gives the arithmetic argument and the acceptance implication.

An accepted result proves that the mathematical nominal objective gap is at
most **1/1,000,000 exactly** at the state mapped from the supplied binary
coordinates. Each binary input value denotes its exact real value. Intensity,
coordinate geometry, and derivative arithmetic are enclosed. The output
semantics are the exact map evaluated at these coordinates, rather than a
separately rounded physical array. Instrument calibration and measurement
uncertainty would require an expanded model with their own enclosures.

## Try a folded optimum

From the package root:

```sh
python theory/validated_stopping/cli.py --x 0 1 0 --y 1 .5 0 .5
```

The default is 128 fractional bits. The JSON result includes an exact upper
endpoint `bound_upper_integer / 2**bits`. The displayed binary64 bound is
rounded upward for readability. Acceptance uses the integer comparison,
not the displayed number. An unresolved denominator or domain yields
`inconclusive`. Supported exponential arguments lie in `[-64,64]`, and a
successful interval evaluation verifies `0 < I <= 8`.

Outer coordinates are handled using the exact inner preimage algebraically,
preserving the original physical output. No optimizer restart occurs. The
implementation evaluates one scalar radial curvature at inner radius at least
1/2; it obtains analytic derivatives of the nominal model and forms no full
head Hessian. Physical gradients are available in this implementation, so
the direct physical methods remain relevant controls.

## Frozen retrospective check

The protocol retains all 9,284 previously inspected real-data warm endpoints.
No observations, histories, optimizer outputs, or stopping thresholds change.
The four precisions are fixed before evaluation. Results are:

| Fractional bits | Certified | Inconclusive | Accurate rejected | One-pass time (s) |
|---:|---:|---:|---:|---:|
| 32 | 8,858 | 3 | 131 | 6.21 |
| 64 | 8,861 | 0 | 128 | 6.38 |
| 128 | 8,861 | 0 | 128 | 7.32 |
| 256 | 8,861 | 0 | 128 | 8.73 |

Every precision accepts zero inaccurate endpoints. A separate 256-bit direct
cone calculation classifies 8,989 accurate and 295 inaccurate endpoints, with
no threshold-overlap intervals. The 64--256-bit decisions reproduce the prior
ordinary-float pointwise decisions. Three 32-bit radius enclosures include
zero in a denominator, so those accurate cases remain inconclusive. All four
runs make 52 scalar curvature evaluations and form zero full head Hessians.
Times describe one CPU pass and exclude the separate reference computation.

## Reproduce and inspect

All deterministic integer endpoint records and fixed source hashes are in
`results/`. Reproduce them in a fresh directory without downloads or packages:

```sh
python theory/validated_stopping/replay.py --output /tmp/stokes-validated-replay
python theory/validated_stopping/independent_checks.py --output /tmp/stokes-arithmetic-checks.json
python theory/validated_stopping/check_cartesian.py /tmp/stokes-cartesian-checks.json
```

The replay compares all endpoint, reference, and stress records byte for byte;
scientific summaries exclude observed times and environment metadata. The
independent arithmetic audit uses exact rational endpoint operations and an
exponential oracle that directly sums at the original argument, avoiding the
implementation's range reduction. The Cartesian check uses exact `I=1` and
builds full derivatives independently to check the radial tolerance formulas.
These checks test interval containment and the Cartesian derivative formulas;
Supplement S21 gives the acceptance argument.

Outward inclusion arithmetic is established; `PRIOR_ART.json` records the
checked Rump reference. This addition makes the stated Stokes bound executable
under explicit nominal-model semantics. The earlier experiment records remain
unchanged.
