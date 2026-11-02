# Find the evidence behind each result

Follow a result from the paper to its implementation and supplied numerical records. Start with the stopping certificate, then trace its recovery identities or inspect the experimental comparisons. The saved outcomes linked below are the paper's reference records.

[Research overview](../README.md) · [Main paper](../paper/arxiv.pdf) · [Supplementary Information](../paper/supplement.pdf) · [Combined reader copy](../paper/complete.pdf)

Before running a study, consult the [reproduction guide](REPRODUCING.md) for its commands, recorded environment and output behavior. Separate studies use separate environments; preserve the supplied records by working in a fresh copy or using the documented output option.

## Stopping guarantees and numerical certification

| Result | Paper location | Implementation and entry points | Supplied evidence |
| --- | --- | --- | --- |
| Uniform and pointwise stopping bounds | Theorem 7; Figure 1; Supplement S20 | [Replay](../theory/stopping_guarantee/replay.py), [certificate formulas](../theory/stopping_guarantee/certificate.py) | [Summary](../theory/stopping_guarantee/results/SUMMARY.json), [all stopping records](../theory/stopping_guarantee/results/) |
| Outward interval certificate | Supplement S21 | [Small-example CLI](../theory/validated_stopping/cli.py), [replay](../theory/validated_stopping/replay.py), [monitor](../theory/validated_stopping/monitor.py), [interval arithmetic](../theory/validated_stopping/arithmetic.py) | [Summary](../theory/validated_stopping/results/SUMMARY.json), [all interval records](../theory/validated_stopping/results/) |

## Recovery identities, sensitivity and acquisition cost

| Result | Paper location | Implementation and entry points | Supplied evidence |
| --- | --- | --- | --- |
| Exact fold recovery and first-order ambiguity | Proposition 1; Figure 2 | [Identifiability checks](../repro/identifiability_checks.py), [fold-gradient recovery](../repro/fold_gradient_recovery.py) | [Identifiability results](../repro/results/identifiability_checks.json), [recovery records](../repro/results/) |
| Mixed/vector recovery and input gains | Propositions 2–3; Corollary 4; Figure 3 | [Mixed recovery](../repro/mixed_recovery_checks.py), [corrected recovery](../repro/corrected_recovery_checks.py), [local curvature](../repro/local_curvature_checks.py), [anisotropic curvature](../repro/anisotropic_curvature_checks.py) | [Mixed](../repro/results/mixed_recovery_checks.json), [corrected](../repro/results/corrected_recovery_checks.json), [local](../repro/results/local_curvature_checks.json), [anisotropic](../repro/results/anisotropic_curvature_checks.json) |
| Quadratic physical-gap bound | Corollary 5; Supplement S4 and S10 | [Calibrated certificate](../repro/calibrated_certificate.py), [boundary check](../repro/boundary_robust_check.py) | [Boundary results](../repro/results/boundary_robust_check.json), [recovery records](../repro/results/) |
| AD acquisition cost and finite precision | Supplement S14; Figure 3 | [AD benchmark](../repro/ad_route_benchmark.py) | [Benchmark records](../repro/results/ad_route_benchmark.json) |

## Fitting, observations and repair controls

| Result | Paper location | Implementation and entry points | Supplied evidence |
| --- | --- | --- | --- |
| Shared-network fitting | Supplement S15 and S23 | [Training and diagnostics](../repro/trained_fold_diagnostic.py), [verification](../repro/verify_trained_fold.py) | [Saved runs](../repro/results/trained_fold/), [diagnostic results](../repro/results/trained_fold_diagnostic.json) |
| Controlled warm starts | Supplement S17; Figure 4 | [Warm-start verification](../repro/warm_start/verification/verify_warm_start.py) | [Verification records](../repro/warm_start/verification/) |
| Noisy Fresnel sequences | Supplement S18; Figure 5 | [Fresnel replay](../application/fresnel_sequence/reproduce.py) | [Results](../application/fresnel_sequence/results.json), [summary](../application/fresnel_sequence/SUMMARY.json) |
| Released observations | Supplement S18; Figure 6 | [Observation replay](../application/real_data/replay.py) | [Inputs](../application/real_data/INPUTS.json), [results](../application/real_data/RESULTS.json), [summary](../application/real_data/SUMMARY.json) |
| Coordinate repair and direct controls | Supplement S19; Table 2 | [Repair replay](../application/chart_repair/replay.py) | [Frozen comparison records](../application/chart_repair/frozen/) |

## Comparing a replay with the supplied records

The saved figure inputs and numerical outcomes, rather than PDF byte identity, are the comparison targets after regeneration. Wall-clock timings are specific to the recorded run and are excluded from exact scientific-summary comparisons where the replay states this explicitly.
