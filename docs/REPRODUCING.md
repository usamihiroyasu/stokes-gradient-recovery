# Reproducing the numerical results

Run commands from the repository root. The original recovery checks use Python 3.12, NumPy 2.3.5, SciPy 1.18.1, and Matplotlib 3.11.2 (`repro/requirements.txt`). The later Fresnel, released-data, coordinate-repair, and stopping checks use Python 3.12.3 and NumPy 2.5.4 with the pinned requirements in `application/real_data/requirements.txt`. Use separate environments when reproducing these recorded versions.

## Recorded environments

The versions in the supplement describe separate runs. They are not a single
shared software stack.

| Study | Python | NumPy | SciPy | PyTorch | Evidence |
| --- | --- | --- | --- | --- | --- |
| S14 AD benchmark and S15 shared MLP | 3.12.14 | 2.3.5 | Not recorded | 2.14.1 | Saved `environment` fields in `repro/results/ad_route_benchmark.json` and `trained_fold_diagnostic.json` |
| S17 SciPy continuation | 3.12.3 | 2.5.4 | 1.18.1 | Not used | `repro/warm_start/verification/environment.json` |
| S17 PyTorch precision control | 3.11.6 | 2.3.5 | Not used | 2.9.1 | `precision_results.json` metadata and `independent_torch_ad.json` in the same verification directory |
| S18--S20 main application and stopping calculations | 3.12.3 | 2.5.4 | 1.18.1 | Not used | Per-study protocols and the clean source-package replay |
| S18--S20 independent AD audits | Not recorded | Not recorded | Not recorded | 2.9.1 | Each audit's recorded PyTorch version; other versions are not inferred from the S17 control |
| S21 interval replay | 3.12.3 | Not required | Not required | Not required | Standard-library arithmetic and replay |

The clean replay environment on 22 October 2026 directly imported Python
3.12.3, NumPy 2.5.4, SciPy 1.18.1, and Matplotlib 3.11.2, with no PyTorch
installed. System-site and user-site packages were disabled. This confirms
that the Python 3.12.3 / NumPy 2.5.4 combination is an executed environment.
The replay retains the saved S14 timings and MLP training records; it does not
recreate that historical environment or remeasure those timings.

## Recovery and sensitivity

```sh
.venv/bin/python repro/identifiability_checks.py
.venv/bin/python repro/corrected_recovery_checks.py
.venv/bin/python repro/local_curvature_checks.py
.venv/bin/python repro/anisotropic_curvature_checks.py
.venv/bin/python repro/mixed_recovery_checks.py
```

Each command writes its report to `repro/results/`. These commands check:

| Check | Cases | Reference |
| --- | ---: | --- |
| Quadratic gap brackets | 274 | Saved primal/dual candidates for cone quadratics |
| Nonquadratic fold recovery | 27 per route | 80-digit Decimal differences |
| Quadratic radius comparison | 30 | First inverse, corrected HVP, fold-formula reuse |
| Nonquadratic vector correction | 7 | Local physical Hessian action |
| Curvature perturbations | 12 | Exact linear error identity |
| Directional gains | 34 | Radial and tangential perturbations |
| Mixed recovery and errors | 30 each | Scalar differences and exact error identity |
| Mixed ambiguity | 8 pairs | Equal restricted observations, unequal gradients |

Decimal differences check numerical consistency. Float64 primal–dual values are numerical brackets rather than outward-rounded interval enclosures. Reporting norms and tolerances are specified in the reports and supplement.

To recompute the underlying cone optimizations before the bracket checks:

```sh
.venv/bin/python repro/calibrated_certificate.py
.venv/bin/python repro/fold_gradient_recovery.py
.venv/bin/python repro/robust_sensitivity.py
```

## Figures

```sh
.venv/bin/python repro/theory_experiments.py
.venv/bin/python repro/boundary_robust_check.py
.venv/bin/python repro/head_geometry.py latex/images
.venv/bin/python repro/diagnostic_figures.py
.venv/bin/python repro/identifiability_figures.py
```

Run `identifiability_figures.py` last: it writes the current main-paper figures. The other generators also supply supplementary figures. Generated PDF metadata may vary while plotted values remain unchanged.

The saved `repro/data/renderer_reference.npz` is a 32 × 32 finite-sampling Mitsuba reference. Its SHA-256 is `7bd03aa4a8c6c6a4384d02f6ff5cb027f240e3475475f61bb0133c301c2db824`. The supplement specifies the analytic sphere scene, samples, seeds, Stokes convention, and target-informed stationary construction. Figure generation reads this array without invoking a renderer.

## Optional automatic-differentiation comparison

```sh
python3 -m venv .ad-venv
.ad-venv/bin/python -m pip install -r repro/requirements-ad.txt
.ad-venv/bin/python repro/ad_route_benchmark.py --check-only
.venv/bin/python repro/ad_table.py
```

The check-only mode recomputes accuracy records without replacing saved timings. Run without `--check-only` to measure new timings on the current machine. `ad_table.py` typesets the saved record.

The recorded experiment uses PyTorch 2.14.1 on an Apple M4 Max CPU, one thread, float32/float64, and 36 intensity/radius states. Four of 144 route evaluations are singular first-inverse cases at the fold; 140 produce finite gradients. Timing includes graph construction, the required physical-curvature query, and gradient recovery. Scalar curvatures in this implementation are contractions of vector HVPs. Supplement S14 gives batch sizes, order rotation, and interquartile reporting.

## Integrity and manuscript build

```sh
shasum -a 256 -c MANIFEST.sha256
.venv/bin/python scripts/prepare_template.py
.venv/bin/python scripts/build_paper.py
```

Verify the supplied manifest before regenerating records or PDFs. Generated outputs are expected to change after regeneration. To use an already downloaded official template, pass `--archive /path/to/template.zip` to `prepare_template.py`; the same archive and per-file hashes are checked.

## Shared-network fitting

Install the AD requirements, then run:

```sh
.ad-venv/bin/python repro/trained_fold_diagnostic.py --check-only
.ad-venv/bin/python repro/verify_trained_fold.py --part head
.venv/bin/python repro/verify_trained_fold.py --part cone
.venv/bin/python repro/training_figures.py
```

Run without `--check-only` to repeat the 20 fixed training runs. The protocol is in `repro/trained_fold_protocol.json`; saved predictions, observations, raw coordinates, weights, and losses are in `repro/results/trained_fold/`. Five seeds and two initializations are separate runs; pixels and checkpoints are not independent replications. Run `training_figures.py` after `identifiability_figures.py` to restore the current main-paper plots from observed AD errors and training records. The prescribed-envelope experiment remains in the saved reports and supplement.

## Physical reflection and released data

The additional studies use Python 3.12.3 with the pinned requirements in
`application/real_data/requirements.txt`. From the extracted repository root:

```sh
python application/fresnel_sequence/reproduce.py --output /tmp/stokes-fresnel-replay
python application/real_data/replay.py --output /tmp/stokes-real-replay
python application/real_data/make_paper_assets.py
```

Choose fresh output directories. The replays preserve supplied records and
compare observation hashes and scientific summaries. The real-data replay
uses included selected observations and needs no image download. The
`verification/verify_warm_start.py` module is an exact copy of the numerical
core used by these application scripts.

For independent acquisition and fitting of the pinned original subset:

```sh
python application/real_data/fetch_subset.py --output /tmp/stokes-real-originals
python application/real_data/fit_subset.py --acquired /tmp/stokes-real-originals --output /tmp/stokes-real-refit
```

The fetcher validates public ZIP byte ranges and checks every selected member
against its recorded SHA-256; no full archive or checkpoint is needed. See
`application/real_data/README.md` for observation units, source layouts,
conditional selections, static versus temporal ordering, and the distinction
between native solver success and physical-monitor acceptance. Raw full images
are not redistributed in this package.

## Frozen coordinate-repair follow-up

This is development on the already inspected released observations and Fresnel
sequences. Inputs, ordering, thresholds and budgets are unchanged; there is no
new data download or training. From the package root, with the pinned
`application/real_data/requirements.txt` dependencies installed:

```sh
python application/chart_repair/replay.py --output /tmp/stokes-chart-replay
```

Use a new empty directory. The replay compares exact scientific summaries,
excluding timing, and logs every internal native solver status, warning and
nonfinite query. See `application/chart_repair/README.md` for the distinction
between continuation histories, paired endpoint correction, monitored
acceptance and native success. Optional independent checks use PyTorch:

```sh
python application/chart_repair/independent_audit.py
```

The existing physical bound is evaluated in float64; it is not outward-rounded
numerical certification. Standard physical controls are retained alongside the
coordinate repairs.

## Finite-tolerance stopping checks

```sh
python theory/stopping_guarantee/replay.py --output /tmp/stokes-stopping-replay
```

This preserves the frozen source results, reproduces 600 objective checks and
1,800 derivative-error cases, and audits all 9,284 previously inspected warm
endpoints without new optimizer fits on those observations. Choose a new empty
directory. Optional checks use PyTorch for independent AD and SymPy for exact
rational identities and limits:

```sh
python theory/stopping_guarantee/audit_torch.py
python theory/stopping_guarantee/symbolic_audit.py
```

See `theory/stopping_guarantee/PRIOR_ART.json` for PDF versus HTML theorem
locators and the scope of the novelty comparison. The main stopping theorem
requires strong convexity, justified model constants and supplied derivative
error budgets; the float64 checks do not establish those budgets.

## Public environment metadata

Machine names and absolute interpreter paths are omitted from the public
environment metadata. Recorded package versions, scientific inputs, and
numerical outcomes are retained.
