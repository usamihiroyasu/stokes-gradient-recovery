# From paper results to executable sources

Run each study with the environment listed in `docs/REPRODUCING.md`. The paths below are relative to the package root. The saved outcomes are the paper's reference records.

| Result | Paper location | Main source or entry point | Frozen evidence |
| --- | --- | --- | --- |
| Exact fold recovery and first-order ambiguity | Proposition 1; Figure 2 | `repro/identifiability_checks.py`, `repro/fold_gradient_recovery.py` | `repro/results/` |
| Mixed/vector recovery and input gains | Propositions 2–3; Corollary 4; Figure 3 | `repro/mixed_recovery_checks.py`, `repro/corrected_recovery_checks.py`, `repro/local_curvature_checks.py`, `repro/anisotropic_curvature_checks.py` | `repro/results/` |
| Quadratic physical-gap bound | Corollary 5; Supplement S4 and S10 | `repro/calibrated_certificate.py`, `repro/boundary_robust_check.py` | `repro/results/` |
| AD acquisition cost and finite precision | Supplement S14; Figure 3 | `repro/ad_route_benchmark.py` | `repro/results/ad_route_benchmark.json` |
| Shared-network fitting | Supplement S15 and S23 | `repro/trained_fold_diagnostic.py`, `repro/verify_trained_fold.py` | `repro/results/trained_fold/` and `repro/results/trained_fold_diagnostic.json` |
| Controlled warm starts | Supplement S17; Figure 4 | `repro/warm_start/verification/verify_warm_start.py` | `repro/warm_start/verification/` |
| Noisy Fresnel sequences | Supplement S18; Figure 5 | `application/fresnel_sequence/reproduce.py` | `application/fresnel_sequence/results.json`, `application/fresnel_sequence/SUMMARY.json` |
| Released observations | Supplement S18; Figure 6 | `application/real_data/replay.py` | `application/real_data/INPUTS.json`, `application/real_data/RESULTS.json`, `application/real_data/SUMMARY.json` |
| Coordinate repair and direct controls | Supplement S19; Table 2 | `application/chart_repair/replay.py` | `application/chart_repair/frozen/` |
| Uniform and pointwise stopping bounds | Theorem 7; Figure 1; Supplement S20 | `theory/stopping_guarantee/replay.py`, `theory/stopping_guarantee/certificate.py` | `theory/stopping_guarantee/results/` |
| Outward interval certificate | Supplement S21 | `theory/validated_stopping/cli.py`, `theory/validated_stopping/replay.py`, `theory/validated_stopping/monitor.py`, `theory/validated_stopping/arithmetic.py` | `theory/validated_stopping/results/` |

The saved figure inputs and numerical outcomes, rather than PDF byte identity, are the comparison targets after regeneration. Wall-clock timings are specific to the recorded run and are excluded from exact scientific-summary comparisons where the replay states this explicitly.
