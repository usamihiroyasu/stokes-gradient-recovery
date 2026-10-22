# Numerical conventions

The theory uses symbolic intensities, radii, metrics, and input-error bounds. The constants below define the retained numerical studies and their implementation guards.

## Geometry and normalization

- The ideal analyzer matrix has rows `(1, cos(2θ), sin(2θ))/2` at `θ = 0, π/4, π/2, 3π/4`. Thus `W = A.T @ A = diag(1, 1/2, 1/2)`. For a unit direction `e`, the normal `n = (-1, e)` satisfies `n.T @ inv(W) @ n = 3`. The divisor `3` in the ideal-metric multiplier and boundary projection comes from this metric.
- Stokes amplitudes in the numerical fields use relative radiance units. The normalizations by squared intensity use those same units throughout each study.
- The radii `1` and `sqrt(3)` are analytic: the first zeros the radial eigenvalue of `DF`; the second zeros the radial eigenvalue of the map-curvature matrix `K`. The latter is a singularity of the curvature-corrected route, while the first-order inverse remains regular there.

## Fitting and incidence

The complete settings are in [`trained_fold_protocol.json`](../repro/trained_fold_protocol.json), with the formulas in Supplement S15.

| Constant | Location | Role |
| --- | --- | --- |
| 32 × 32 pixels | Both input fields | Retained field resolution; 1,024 fitting samples per run |
| Widths 4–32–32–3 | Shared MLP | Four analyzer inputs, two hidden layers, three head coordinates |
| 0.01 | Gaussian analyzer noise | Standard deviation in relative measurement units |
| 0.003; 1,500 updates | Adam | Fixed learning rate and fitting budget |
| (0.9, 0.999); 1e-8 | Adam | Moment parameters and denominator offset; zero weight decay |
| 0.2 and 3 | Initialization | Multipliers on final-layer polarization weights for standard and broad initialization |
| Seeds 0–4; offset 1,000 | Random generators | Network seeds; separate observation-noise generator seed |
| 1e-6 | Intensity head | Positive offset added to softplus |
| 1e-3, 1e-4, 1e-5 | Fold incidence | Neighborhood widths in raw radius `abs(r - 1)` |
| −1e-7 | Fold incidence | Upper cutoff on `c_r / I**2` for a negative-curvature candidate |
| 1e-5 | Head stationarity | Upper cutoff on the normalized independent-head gradient norm |
| 0.001, 0.01, 0.05, 0.1 | Radial trials | Signed independent-head radius offsets at fixed intensity |

The stationarity denominator is `max(I**2, 1e-12)`. The curvature cutoff uses `I**2` directly. An auxiliary clean-reference error divides by `I_target**2 + 1e-12`. The `1e-12` terms prevent zero denominators; they do not alter the mathematical recovery identities.

## Numerical guards and display

- The training code allows absolute cone/intensity/DoLP assertion slack `2e-6` for float32 rounding. Diagnostics compare gap and bound with absolute allowance `2e-12` in float64. These are implementation assertions.
- The best-drop/bound diagnostic ratio uses denominator `max(bound, 1e-300)` to avoid division by zero. It does not determine candidate membership.
- SLSQP tolerances, iteration limits, finite-difference steps, decimal working precision, and comparison allowances are specified by study in Supplement S5 and S8–S13. The SLSQP objective tolerance is `1e-13`; the boundary classification tolerance for its non-apex KKT residual is `1e-7`.
- Logarithmic plots clip small displayed losses at the values stated in their captions. Saved numerical records retain the original values.
- The coefficients of the three nonquadratic test functions and synthetic image fields specify those functions. They are not fitted physical parameters. Figure 1 selects three signs of its normal-gradient coefficient to illustrate the classification.

## Coordinate repair and stopping

The empirical fold gate uses `c_r / I**2`, a curvature normalized by the
quadratic-loss intensity scale. The stopping theorem uses `c_r / I`, whose
units match the physical-gradient tolerance `nu`. These are different tests.

| Value | Role and source |
| --- | --- |
| `1e-6` | Registered nominal objective-gap threshold in the application audits; the interval monitor compares against the exact fraction `1/1000000`. It is an experimental accuracy target, not a fitted model constant. |
| `abs(r-1) <= 1e-3`, `c_r/I**2 < -1e-7` | Registered local fold gate; it has incomplete coverage away from the fold. |
| Radial offsets `-0.01`, `+0.01` | Registered restart candidates, selected together with the current point by objective value. |
| Step `1`, maximum `64` physical updates | The step uses the ideal quadratic's largest Hessian eigenvalue `1`; `64` is the fixed iteration budget. Acceptance still requires the gap-bound test. |
| `mu=L=1/2` | Exact smallest Hessian eigenvalue and polarization curvature for the ideal four-analyzer quadratic. |
| `Ibar=8` | Chosen upper intensity domain for the retrospective stopping study; the monitor checks membership. It is not a camera-calibration constant. |
| `r=1/2`, `epsilon <= K0/8`, `K0=Ibar*L` | Proof-band choices: `h=(epsilon/K0)^(1/3) <= 1/2` permits the near-fold estimates in S20. |
| `52`, `104`, `6` | The uniform coefficient rounds `5/2+48+1=51.5` upward. The target-gap rule uses `104=2*52` and `6=2*3` to allocate half the gap budget to each term. |
| `32`, `64`, `128`, `256` fractional bits | Fixed interval-precision settings; all are reported, including inconclusive results. |
| Exponential range `[-64,64]`, reduced argument at most `1/8`, Taylor degree `48` | Declared implementation limits and enclosure choices. The next term and ratio bound `t/50` supply the positive remainder; unsupported inputs remain inconclusive. |

For the target-gap example with `mu=L=1/2`, `Ibar=8`, and `tau=1e-6`, rounding
the exact theoretical thresholds downward gives `epsilon <= 2.886e-7` and
`nu <= 2.886e-4`. Rounding a sufficient
upper threshold upward would not preserve the sufficient implication.
