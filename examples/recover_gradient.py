"""Recover a quadratic physical gradient from the observation sets in Table 1."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repro"))
from identifiability_checks import raw


def main():
    intensity = 1.2
    direction = np.array([0.6, 0.8])
    metric = np.array([[1.4, 0.1, -0.05], [0.1, 0.8, 0.02], [-0.05, 0.02, 0.6]])
    target = np.array([0.9, 0.15, -0.2])
    radial = np.outer(direction, direction)
    tangent = np.eye(2) - radial

    for radius in [0.99, 1.0, 1.01]:
        phi, a, t, jacobian, curvature, truth, b, j, v = raw(
            intensity, radius, direction, target, metric
        )
        c = float(direction @ v)
        hrr = float(direction @ metric[1:, 1:] @ direction)
        kappa = float(direction @ curvature @ direction)
        m = (c / intensity - intensity * a * a * hrr) / kappa
        mixed = np.r_[b - phi * m, tangent @ j / (intensity * t) + direction * m]

        gp = np.linalg.solve(
            curvature,
            v / intensity - intensity * a * jacobian @ metric[1:, 1:] @ direction,
        )
        vector = np.r_[b - phi * direction @ gp, gp]
        recovered = {"mixed": mixed, "vector": vector}
        if radius == 1.0:
            recovered["fold scalar"] = np.r_[b + c / intensity, (j - c * direction) / intensity]
        else:
            gp = radial @ j / (intensity * a) + tangent @ j / (intensity * t)
            recovered["first inverse"] = np.r_[b - phi * direction @ gp, gp]
        for route, estimate in recovered.items():
            np.testing.assert_allclose(estimate, truth, rtol=1e-12, atol=1e-12)
            print(f"r={radius:.2f}  {route:13s}  error={np.linalg.norm(estimate - truth):.3e}")


if __name__ == "__main__":
    main()
