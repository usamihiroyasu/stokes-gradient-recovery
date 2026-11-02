# Sequential stopping verification

The fixed two-frame study asks whether a solver's head-coordinate acceptance
agrees with the physical cone objective gap after it inherits the preceding
raw estimate. The preceding frame is fitted by the same solver, starting
from a fixed head coordinate independent of the truth.

The 60 source conditions share noise across settings. Three next-frame angle
changes, three SciPy solvers, and three tolerances give 1,620 continuations.
A separate true float32/64 PyTorch LBFGS control gives 360 continuations.
They are controlled comparisons, not independent scenes or a population rate.

## Records

All paths below are relative to `repro/warm_start/`.

- `strategy/warm_start_probe.json`: frozen observation vectors and condition metadata.
- `strategy/raw_warm_start_probe.json`: preliminary same-solver reference.
- `verification/PROTOCOL.json` and `PROTOCOL_RECEIPT.json`: fixed settings and code hashes before optimization.
- `verification/main_results.json` and `precision_results.json`: endpoints, flags, gradients, gaps, warnings, and optimizer query counts.
- `verification/comparison.csv` and `precision_comparison.csv`: complete grid summaries.
- `verification/audited_summary.json`: finite and nonfinite errors separated.
- `verification/failure_records.json`: native SciPy failures and nonfinite calls; PyTorch nonacceptances remain in `precision_results.json`.
- `verification/COST_ACCOUNTING.json`: actual optimizer queries, analysis overhead, terminal checks, and gate requirements.
- `verification/cone_slsqp_audit.json`: independent physical-coordinate minimum checks for 180 observation vectors.
- `verification/independent_torch_ad.json` and `preflight_derivative_audit.json`: derivative checks at 60 frozen endpoints.
- `verification/environment.json`: Air hardware, Python, library versions, protocol hash, and run duration.

## Reproduction

Copy this directory before rerunning: scripts write result files next to the
verification code. The two pinned environments are listed in
`requirements-scipy.txt` (Python 3.12.3) and `requirements-torch.txt` (Python
3.11.6). With the corresponding interpreter/environment, run from the project
root:

```sh
python repro/warm_start/verification/precision_and_ad.py --ad-only
python repro/warm_start/verification/verify_warm_start.py
python repro/warm_start/verification/precision_and_ad.py
python repro/warm_start/verification/audit_and_summarize.py
python repro/warm_start/verification/cost_accounting.py
python repro/warm_start/figure.py --input repro/warm_start/verification --output latex/images/warm_start_audit
```

The solver code does not import the preliminary strategy modules. It evaluates
the head derivatives using Cartesian component formulas. PyTorch acquires
derivatives independently and retains the native tensor precision during
optimization; its diagnostic controller and reference gaps use float64.

## Interpretation

The native SciPy success flag and PyTorch gradient acceptance are distinct
from gap accuracy. A false success is acceptance with cone gap above 1e-6.
Failures are not removed, and nonfinite numbers are serialized as null with
their flags, statuses, and counts. The audited summary counts them explicitly.

The fixed local gate requires both a radius distance at most 1e-3 and
normalized radial curvature below -1e-7. It detects the nominal near-fold
false successes but misses other failures and many loose-tolerance endpoints.
The physical gap bound includes interior support slack. Its float64 empirical
acceptance counts are not outward-rounded numerical certificates.

The restricted-flow proposition assumes an ideal fixed metric, cone-interior
unconstrained estimate, exact fold initialization, and ordinary raw Euclidean
continuous flow. Its antipodal exception is explicit. No fold invariance is
claimed for discrete GD, Adam, BFGS, or LBFGS.

The query ledger distinguishes the Hessians actually evaluated for complete
coverage logging from those required after applying a radius guard first.
Function/gradient pairs and Hessian queries are not equivalent units of work;
the counts do not establish a wall-clock speed advantage.
