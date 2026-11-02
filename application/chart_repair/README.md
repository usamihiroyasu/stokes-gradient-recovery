# Coordinate repair on the frozen Stokes observations

This follow-up was designed after inspecting the preceding real-data audit.
It is development on known data, not independent validation or a new imaging
benchmark. The 4,642 released observations and 984 Fresnel frames are unchanged.
The algorithm, thresholds and controls were frozen before their new outcomes.

## Findings

Counts below are BFGS/L-BFGS-B. Each head arm carries its own previous endpoint.
An inaccurate output has nominal objective gap above 1e-6, including outputs
that the physical stopping monitor rejects.

| Strategy | Released inaccurate | Fresnel inaccurate |
| --- | ---: | ---: |
| Raw warm start | 127/168 | 9/7 |
| Cold start | 0/14 | 0/0 |
| Switch outer coordinates inward before each warm solve | 9/16 | 12/10 |
| Audited cold restart | 0/2 | 0/0 |
| Audited coordinate repair | 7/22 | 0/0 |
| Coordinate repair plus physical fallback | 0/0 | 0/0 |
| Standalone physical projected gradient (one algorithm) | 0 | 0 |
| Direct cone reduction (one algorithm) | 0 | 0 |

The switch is w -> w/(w dot w), applied only outside the unit circle. It preserves
the output and amplifies the polarization head-gradient norm by r^2, while
leaving its condition number and fold unchanged. Fresh solver calls reset
quasi-Newton history. The audited chart restarts at most once on a physical-bound
rejection. Near the fold it uses the previously fixed negative-curvature gate
and radial trial rule. The guarded arm then falls back to standard physical
projected gradient, at most 64 updates, if still rejected.

On the released sequences the guarded arms use 16/25 outer restarts and 2/15
physical fallbacks, requiring 3/22 projected updates. They require no Hessian.
Their physical audits acquire 4,663/4,704 physical gradients. On Fresnel each
guarded solver uses 7 full Hessians and 21 trial FG pairs, without physical
fallback. No final inaccurate output is accepted by a monitored arm.

The paired audit starts all procedures from the same 9,284 historical warm
endpoints, including all non-rejections. Coordinate repair alone corrects 249
of the original 295 false successes, leaving 7/39. Physical fallback fixes those
remaining 46. Cold restarting leaves 0/5. These counts differ from continuation
because sequential interventions alter later initializations.

The straightforward physical controls remain stronger for this exposed
quadratic. Standalone projected gradient has no inaccurate outputs and uses
9,575 updates over the released data. Its single-pass observed time is 0.203 s,
versus 4.64/1.35 s for the guarded head methods and 0.0219 s for direct cone
reduction. The protocol does not support statistical timing or generalization
claims. The coordinate intervention is a partial repair for an existing head,
not evidence that a new head solver is preferable to direct physical solving.

## Files and reproduction

- `frozen/PROTOCOL.json`: frozen code/input hashes, controls and budgets.
- `frozen/RESULTS.json`: all 78,764 sequential output records.
- `frozen/PAIRED.json`: all 27,852 paired procedure outputs.
- `frozen/SUMMARY.json`: per-stratum and aggregate results, timings and counts.
- `run.py`: immutable registered algorithm; verifies its code and input hashes.
- `replay.py`: exact scientific-summary replay, also logging every internal
  native solver status, warning and nonfinite count.
- `independent_audit.py`: PyTorch sheet symmetry and independent physical steps
  on all 295 historical failures plus 32 fixed controls.

From the source-package root, install the requirements already specified for
the existing real-data audit, then run:

```sh
python application/chart_repair/replay.py --output /tmp/stokes-chart-replay
```

Choose a new empty output directory. Inputs come from the included selected
observations and prior records in `application/real_data/` and
`application/fresnel_sequence/`; no image download or training is required.
Scientific summaries must agree exactly; measured times are excluded.

Optional independent audit, using a Python environment with PyTorch:

```sh
python application/chart_repair/independent_audit.py
```

The measured monitoring bounds use float64, without outward-rounded numerical
certification. In exact arithmetic acceptance by the existing upper bound
implies nominal gap at most 1e-6. A fixed iteration budget alone does not imply
acceptance; unresolved results remain rejected. No imaging ground-truth,
camera-calibration, population-incidence or unseen-scene claims are made.
