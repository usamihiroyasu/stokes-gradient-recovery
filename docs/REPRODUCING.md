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
```

The check-only mode recomputes accuracy records without replacing saved timings. Run without `--check-only` to measure new timings on the current machine.

The recorded experiment uses PyTorch 2.14.1 on an Apple M4 Max CPU, one thread, float32/float64, and 36 intensity/radius states. Four of 144 route evaluations are singular first-inverse cases at the fold; 140 produce finite gradients. Timing includes graph construction, the required physical-curvature query, and gradient recovery. Scalar curvatures in this implementation are contractions of vector HVPs. Supplement S14 gives batch sizes, order rotation, and interquartile reporting.

## Integrity and manuscript build

The primary reader copy is [the arXiv-generated draft](../paper/arxiv.pdf). Its exact source bundle and local build commands are in [manuscript/arxiv](../manuscript/arxiv/README.md). The command below builds the separately preserved journal-format main and supplementary PDFs.

From the package root:

```sh
python3 validate_package.py
python3 manuscript/build.py
```

The manuscript build requires a current TeX Live installation with the packages listed in `manuscript/README.txt`. The included class and scientific figures are used; no external journal template download is needed. Main and supplementary sources are separate, cross-referenced files. Keep the generated PDFs in the same directory.

## Shared-network fitting

Install the AD requirements, then run:

```sh
.ad-venv/bin/python repro/trained_fold_diagnostic.py --check-only
.ad-venv/bin/python repro/verify_trained_fold.py --part head
.venv/bin/python repro/verify_trained_fold.py --part cone
.venv/bin/python repro/training_figures.py
```

Run without `--check-only` to repeat the 20 fixed training runs. The protocol is in `repro/trained_fold_protocol.json`; saved predictions, observations, raw coordinates, weights, and losses are in `repro/results/trained_fold/`. Five seeds and two initializations are separate runs; pixels and checkpoints are not independent replications. Run `training_figures.py` after `identifiability_figures.py` to restore the current main-paper plots from observed AD errors and training records. The prescribed-envelope experiment remains in the saved reports and supplement.

## Controlled warm-start runs

The saved S17 inputs, protocol, reference outcomes and checks are under
`repro/warm_start/`. Use the S17 SciPy environment listed above. From the
package root in a fresh working copy:

```sh
python3 repro/warm_start/verification/verify_warm_start.py
```

This script writes results in that copy's verification directory. Preserve the
distributed reference records elsewhere before running it. The separate
root-level `verification/verify_warm_start.py` is the numerical core imported
by the application studies; use the `repro/warm_start/` entry point to rerun S17.

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

## Outward-rounded interval replay

S21 uses the Python standard library and the supplied observation records.
From the package root, select a new empty output directory:

```sh
python3 theory/validated_stopping/replay.py --output /tmp/stokes-interval-replay
```

The replay checks byte identity of the deterministic endpoint, reference and
stress records and compares scientific summaries. Runtime and environment
fields are excluded from that comparison. See
`theory/validated_stopping/README.md` for the scope of the interval certificate.

## Environment metadata

Recorded package versions, scientific inputs and numerical outcomes belong to the individual studies. They do not describe a single universal environment.

New runs may record a local hostname or interpreter path. Keep complete logs
locally; review those fields before sharing a copy. Retain the software
versions, timing conditions, code hashes and numerical records needed to
interpret the experiment.

## Optional symbolic audit

The exact-rational symbolic identities additionally use SymPy 1.14.0:

```sh
python -m pip install sympy==1.14.0
python theory/stopping_guarantee/symbolic_audit.py
```

## Main stopping-certificate figure from saved records

This step only plots supplied records. It does not rerun a solver or modify the supplied observations.

```sh
python3 -m venv .figure-venv
.figure-venv/bin/python -m pip install -r repro/requirements-figures.txt
.figure-venv/bin/python repro/plot_stopping_certification.py --root . --output NEW_FIGURE_DIRECTORY
```

Choose a directory that does not yet exist. The output includes the vector PDF and a hash/count receipt. The reference nominal gap is the midpoint of a stored 256-bit enclosure and is used only as the horizontal plotting coordinate. Display floors and unavailable-bound markers are specified in the manuscript caption.
