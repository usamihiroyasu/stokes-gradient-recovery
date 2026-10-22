# Finite-tolerance stopping on an inner Stokes chart

For a C2, mu-strongly convex physical objective, a rational Stokes state with
0 < I <= Ibar and 0 <= r <= 1 admits the following conditional guarantee.
Normalize the head tests as

    epsilon >= max(abs(partial_a ftilde / Iprime), norm(grad_w ftilde) / I).

At r >= 1/2 require only one radial curvature lower test c_r >= -I*nu and an
upper bound L >= mu on e^T H_pp e at the evaluated physical point. Curvature
is not needed below r=1/2. Set B=Ibar*L. If epsilon <= B/8, then

    physical objective gap <= 52*B^(2/3)*epsilon^(4/3)/mu + 3*nu^2/mu.

The coefficient 52 is conservative. The powers 4/3 and 2 cannot be replaced
uniformly by larger powers in this class. The proof, target-gap thresholds,
tighter pointwise formulas, input-error model and counterexamples are in the
main stopping section and Supplement S20. All earlier experiments are kept.

This is a specialization of established chain-rule and dual-residual ideas.
Approximate transfer is already studied by Ding--Wright; KL exponent 3/4 and
the associated gap power 4/3 are established phenomena in Hadamard models.
Here the Stokes specialization gives an explicit uniform bound from one radial
direction and supplied error budgets. `PRIOR_ART.json` records the
primary-source comparison and its limits.

## Error model and limitations

`certificate.py` takes supplied budgets on normalized intensity derivative,
polarization head-gradient norm and signed radial curvature. Geometry and
model constants must be justified separately. Its float64 evaluation is not
outward-rounded numerical certification; observed float discrepancies do not
establish valid error budgets. For nonconvex physical objectives the global
gap theorem does not apply. No finite-iteration convergence of a native
solver or runtime advantage is claimed.

The domain restrictions are substantive. At the exact fold a first-gradient
test alone misses positive gaps. On an unbounded outer sheet both normalized
first derivatives and negative radial curvature can tend to zero with gap
approaching 1/8. For vanishing exponential intensity, raw first/second tests
can tend to zero with gap approaching 1/2. Inner-sheet conversion requires
re-evaluating derivatives before using the theorem.

## Fixed checks

`results/PROTOCOL.json` freezes 576 quadratics, 24 strongly convex quartics,
three supplied derivative-error budgets and all 9,284 previously inspected
real-data warm endpoints. There are no new optimizer-performance conditions
or adaptive grids. All 600 physical reference solves succeed. The largest
computed gap above a bound is 1.24e-14, with reference feasibility residual
as low as -9.64e-14. This numerical reference error is retained.

Independent PyTorch derivatives check all 600 objectives. Ninety-digit Decimal
records check the sharpness and failure families. `symbolic_audit.py` verifies
the rational identities and asymptotic coefficients with exact SymPy algebra.
The symbolic identities and numerical checks accompany the written proof.

On the known real-data endpoints, 463 outer coordinates are moved inward and
derivatives are recomputed. The pointwise rule accepts 8,861, rejects all 295
historical false successes, and additionally rejects 128 accurate endpoints.
The coarse uniform bound alone accepts 980. This retrospective calculation
uses zero supplied error budgets in float64 and a full head Hessian at every
endpoint; it is neither independent generalization nor a cheaper monitor than
the existing physical-gradient control.

## Reproduce without downloads

From the source-package root, with the NumPy/SciPy requirements of the prior
real-data audit installed:

```sh
python theory/stopping_guarantee/replay.py --output /tmp/stokes-stopping-replay
```

Choose a new empty directory. Input hashes are checked, scientific summaries
are compared, and all AD input arrays must be identical. The source `results/`
is preserved. Optional independent checks require PyTorch and SymPy:

```sh
python theory/stopping_guarantee/audit_torch.py
python theory/stopping_guarantee/symbolic_audit.py
```
