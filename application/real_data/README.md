# Released-observation stopping audit

This bounded study evaluates solver termination for the nominal four-analyzer quadratic on released polarization data. The data selection and numerical settings were fixed before observing solver outcomes. It includes static images/statistical means and recorded bursts, with their different interpretations preserved.

## Sources and selected data

- [DeepSfP data documentation](https://github.com/UCLA-VMG/DeepSfP/blob/master/data/README.md): indoor test objects `box_f`, `father_christmas_f`, and `horse_l`, from its linked archive. Four-angle images and foreground masks are used. Surface normals are not Stokes truth. The selected archive and current repository documentation give different complete split totals; exact members and hashes identify this subset.
- [PolarNS / PolarBurstSR](https://github.com/KAIST-VCLAB/polarns): the lexicographically first object and scene statistics (`241015104249`, `241021103114`) and the first three real-data test crops (`0000000` through `0000002`). The statistical means average 10,000 and 1,017 captures; no new noise is synthesized. All 14 raw PNG frames are used in each burst crop. The authors' `local_burstM_real.py` identifies this test directory as real data, and `data/dataset.py` defines the sensor unpacking used here.

The original multi-gigabyte archives were read through validated public HTTP byte ranges. Every selected member has a matching local ZIP header, verified CRC32, SHA-256, and a recorded byte range. Total selected uncompressed payload is about 189 MB; no full archive or pretrained model was downloaded. Full original images and maps are not included in this source package. The frozen selected observation vectors and coordinates suffice to replay the numerical audit.

`PROTOCOL.json`, `ACQUISITION_PLAN.json`, `DATA_SCHEMA.json`, and `sources/*_ACQUISITION.json` specify the units, layouts, eligibility, source versions, and exact selection. Static samples have 128 seeded uniform pixels plus 64 high-observed-DoLP pixels per image. Bursts have a fixed sensor grid and eight high-DoLP positions per color plane, selected only on frame zero. Uniform and enriched selections may overlap and are reported separately. Static samples follow a prescribed spatial scan; burst tracks use fixed sensor coordinates without registration. They do not establish independent-scene prevalence or tracking of one material point.

## Results

The gap threshold is 1e-6 under the nominal ideal analyzer quadratic. This is optimizer accuracy for that objective, not noise-free Stokes accuracy, camera calibration, denoising, super-resolution, or surface-normal performance. Each row below reports entries per solver/arm, with BFGS/L-BFGS-B count pairs.

| Subset | Entries | Cold native false successes | Warm native false successes | Detected by local curvature gate | Rejected by physical bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSfP | 576 | 0/0 | 0/0 | 0/0 | 0/0 |
| PolarNS means | 384 | 0/7 | 111/64 | 0/0 | 111/64 |
| PolarBurstSR | 3,682 | 0/7 | 16/104 | 0/0 | 16/104 |

All 37,136 fits have native success, including the inaccurate endpoints. No warnings or nonfinite oracle calls occur. The unchanged local gate `abs(r-1)<=1e-3 and c_r/I^2<-1e-7` has no interventions. Curvature-audited branches therefore retain the warm-start false-success counts. The real-data result does not establish coverage or an advantage for that local gate.

The physical bound rejects all 295 raw warm-start false successes and has no inaccurate acceptance across the recorded arms. The physical-audited restart branch still has 68/64 inaccurate native successes in PolarNS means and 16/99 in bursts. Their final physical bounds reject them: rejecting an endpoint and correcting it with one radial restart are distinct outcomes. Large-radius outer-sheet states and small intensities occur among these endpoints. The first-order inverse's polarization gain `1/(I*min(abs(a_r), t_r))` explains why reliable head stopping need not follow from avoiding a small fold neighborhood.

`SUMMARY.json` and `RESULTS.json` retain separate uniform/enriched counts, endpoint states, bounds, native statuses, and query counts. `PAPER_AGGREGATES.json` generates the main-paper table. The diagnostic script evaluates extra Hessians for coverage and final checks. A deployed physical-gradient bound requires zero Hessians; the curvature controller requires them only after its radius guard. These query types and optimizer-only timings do not establish end-to-end speed rankings.

## Independent checks

- All 29 selected physical-coordinate SLSQP cone checks succeed; maximum objective discrepancy is 1.31e-18.
- PyTorch float64 independently checks all 309 cold/warm false successes plus 32 controls. Maximum head-gradient/Hessian/physical-gradient discrepancies are 7.77e-18, 8.89e-16, and 1.39e-17.
- Replaying the frozen selected observations reproduces the input hash and every reported scientific summary. Timing is excluded from comparisons.

The physical-bound acceptance checks use float64 arithmetic and are not outward-rounded certificates. `INDEPENDENT_AD_AUDIT.json` and `CONE_ORACLE_AUDIT.json` contain the checks.

## Reproduction

From the extracted source-package root, use Python 3.12.3 with the pinned NumPy/SciPy/Matplotlib requirements:

```sh
python -m pip install -r application/real_data/requirements.txt
python application/real_data/replay.py --output /tmp/stokes-real-replay
python application/real_data/make_paper_assets.py
```

The first run needs only the included selected observations and numerical core in `verification/verify_warm_start.py`. Select an empty output directory; reference results are preserved. `run_real.py` intentionally refuses to overwrite its registered protocol.

To verify the selected original image/statistic payload independently, use a fresh directory:

```sh
python application/real_data/fetch_subset.py --output /tmp/stokes-real-originals
python application/real_data/fit_subset.py --acquired /tmp/stokes-real-originals --output /tmp/stokes-real-refit
```

The fetcher verifies all selected members against their pinned SHA-256 values. It reads ZIP directories and selected payload ranges only. If source files change or range requests are not honored, it stops instead of accepting different data or downloading whole archives. Independent AD checks additionally require PyTorch 2.9.1 in a compatible environment and run with `python application/real_data/audit_ad.py`.
