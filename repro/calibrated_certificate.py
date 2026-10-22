"""Independent finite-case check of the folded-head SPD-metric certificate.

No random numbers, fitted data, external API, renderer, or neural training.
The QP optimum is computed independently by SciPy SLSQP in Stokes coordinates.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import scipy

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'mpl'))


def analyzer_metric(angles_deg, gain, efficiency, variance):
    theta = np.deg2rad(angles_deg)
    gain, efficiency, variance = map(np.asarray, (gain, efficiency, variance))
    a = 0.5 * gain[:, None] * np.column_stack(
        [np.ones(len(theta)), efficiency * np.cos(2 * theta), efficiency * np.sin(2 * theta)]
    )
    w0 = a.T @ (a / variance[:, None])
    scale = 2.0 / np.trace(w0)
    return a, variance / scale, w0 * scale


def raw_value_gradient(raw, z, w):
    intensity = np.exp(raw[0])
    v = raw[1:]
    d = 1.0 + v @ v
    f = 2.0 * v / d
    df = 2.0 * np.eye(2) / d - 4.0 * np.outer(v, v) / d**2
    s = intensity * np.r_[1.0, f]
    jac = np.column_stack([s, intensity * np.vstack([np.zeros((1, 2)), df])])
    residual = s - z
    return 0.5 * residual @ w @ residual, jac.T @ (w @ residual)


def slack(s):
    return float(s[0] - np.linalg.norm(s[1:]))


def slack_jac(s):
    r = np.linalg.norm(s[1:])
    return np.r_[1.0, -s[1:] / r] if r > 1e-15 else np.array([1.0, 0.0, 0.0])


def solve_qp(z, w):
    def objective(s):
        residual = s - z
        return float(0.5 * residual @ w @ residual)

    res = minimize(
        objective,
        np.array([1.5, 0.0, 0.0]),
        jac=lambda s: w @ (s - z),
        method='SLSQP',
        constraints=[{'type': 'ineq', 'fun': slack, 'jac': slack_jac}],
        options={'ftol': 1e-13, 'maxiter': 1000, 'disp': False},
    )
    p = res.x
    g = w @ (p - z)
    h = slack(p)
    if h <= 1e-7:
        outward_normal = -slack_jac(p)
        multiplier = max(0.0, -float(g @ outward_normal) / float(outward_normal @ outward_normal))
    else:
        outward_normal = np.zeros(3)
        multiplier = 0.0
    stationarity = float(np.linalg.norm(g + multiplier * outward_normal, ord=np.inf))
    return res, {
        'success': bool(res.success), 'status': int(res.status), 'message': str(res.message),
        'iterations': int(res.nit), 'solution': p.tolist(), 'objective': objective(p),
        'cone_slack': h, 'cone_violation': max(0.0, -h),
        'stationarity_inf': stationarity, 'complementarity_abs': abs(multiplier * h),
        'dual_multiplier': multiplier,
    }


def main():
    specs = [
        ('ideal_uncorrelated', [0, 45, 90, 135], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]),
        ('nonideal_angles', [0, 37, 91, 143], [1.03, .97, 1.01, .99], [.94]*4, [1, .8, 1.1, 1.3]),
        ('gain_and_efficiency', [3, 44, 86, 132], [1.15, .88, 1.04, .94], [.91, .95, .89, .97], [1.2, .7, 1.4, .9]),
        ('heteroscedastic', [-5, 40, 95, 139], [.93, 1.08, .96, 1.02], [.90, .96, .93, .88], [.6, 1.7, .85, 1.2]),
    ]
    intensity = 1.2
    directions = [0.0, .7, 1.9]
    alphas = [-.6, -.1, .01, .1, .4, 1.0, 2.0]
    fd_gradient_step = 1e-6
    fd_curvature_step = 1e-4
    matrices, records = [], []
    for name, angles, gain, efficiency, variance in specs:
        a, effective_variance, w = analyzer_metric(angles, gain, efficiency, variance)
        eigenvalues = np.linalg.eigvalsh(w)
        assert eigenvalues.min() > 0
        matrices.append({
            'name': name, 'angles_deg': angles, 'gain': gain, 'efficiency': efficiency,
            'effective_observation_variance': effective_variance.tolist(),
            'A': a.tolist(), 'W': w.tolist(), 'eigenvalues': eigenvalues.tolist(),
            'condition_number': float(eigenvalues.max() / eigenvalues.min()),
            'max_off_diagonal_abs': float(np.max(abs(w - np.diag(np.diag(w))))),
        })
        for angle in directions:
            e = np.array([np.cos(angle), np.sin(angle)])
            normal = np.r_[-1.0, e]
            s = intensity * np.r_[1.0, e]
            raw = np.r_[np.log(intensity), e]
            radial_direction = np.r_[0.0, e]
            inverse_normal = np.linalg.solve(w, normal)
            coefficient = float(normal @ inverse_normal)
            for alpha in alphas:
                z = s - alpha * inverse_normal
                value, gradient = raw_value_gradient(raw, z, w)
                fd_gradient = np.array([
                    (raw_value_gradient(raw + fd_gradient_step * np.eye(3)[i], z, w)[0]
                     - raw_value_gradient(raw - fd_gradient_step * np.eye(3)[i], z, w)[0])
                    / (2 * fd_gradient_step) for i in range(3)
                ])
                gp = raw_value_gradient(raw + fd_curvature_step * radial_direction, z, w)[1]
                gm = raw_value_gradient(raw - fd_curvature_step * radial_direction, z, w)[1]
                curvature_fd = float(radial_direction @ (gp - gm) / (2 * fd_curvature_step))
                curvature_energy_fd = float(
                    (raw_value_gradient(raw + fd_curvature_step * radial_direction, z, w)[0]
                     - 2 * value
                     + raw_value_gradient(raw - fd_curvature_step * radial_direction, z, w)[0])
                    / fd_curvature_step**2
                )
                curvature = -intensity * alpha
                res, qp = solve_qp(z, w)
                gap = value - qp['objective']
                bound = coefficient / (2 * intensity**2) * max(-curvature, 0.0)**2
                z_feasible = slack(z) >= -1e-12
                expected_equality = alpha > 0 and z_feasible
                record = {
                    'matrix': name, 'direction_radians': angle, 'alpha': alpha,
                    'I': intensity, 'e': e.tolist(), 's_fold': s.tolist(), 'z': z.tolist(),
                    'z_cone_slack': slack(z), 'z_feasible': z_feasible,
                    'full_raw_gradient_inf': float(np.linalg.norm(gradient, ord=np.inf)),
                    'raw_gradient_fd_error_inf': float(np.max(abs(gradient - fd_gradient))),
                    'rho_exact': curvature, 'rho_gradient_fd': curvature_fd,
                    'rho_energy_fd': curvature_energy_fd,
                    'rho_gradient_fd_error': abs(curvature_fd - curvature),
                    'rho_energy_fd_error': abs(curvature_energy_fd - curvature),
                    'fold_objective': float(value), 'qp': qp, 'actual_gap': float(gap),
                    'certificate_bound': float(bound), 'bound_minus_gap': float(bound - gap),
                    'expected_equality': expected_equality,
                    'equality_abs_error': abs(gap - bound) if expected_equality else None,
                    'global_fold_solution_error': float(np.max(abs(res.x - s))) if alpha <= 0 else None,
                }
                records.append(record)

    tests = {
        'all_spd': all(min(m['eigenvalues']) > 0 for m in matrices),
        'three_non_diagonal_metrics': sum(m['max_off_diagonal_abs'] > 1e-6 for m in matrices) == 3,
        'ideal_metric_exact': bool(np.allclose(matrices[0]['W'], np.diag([1,.5,.5]), atol=1e-15)),
        'all_solver_success': all(r['qp']['success'] for r in records),
        'all_solver_cone_feasible': max(r['qp']['cone_violation'] for r in records) < 1e-8,
        'all_solver_kkt_stationarity': max(r['qp']['stationarity_inf'] for r in records) < 2e-6,
        'all_solver_complementarity': max(r['qp']['complementarity_abs'] for r in records) < 1e-8,
        'full_raw_stationarity': max(r['full_raw_gradient_inf'] for r in records) < 1e-12,
        'raw_gradient_fd': max(r['raw_gradient_fd_error_inf'] for r in records) < 1e-7,
        'radial_curvature_gradient_fd': max(r['rho_gradient_fd_error'] for r in records) < 1e-6,
        'radial_curvature_energy_fd': max(r['rho_energy_fd_error'] for r in records) < 2e-6,
        'global_minimum_for_nonpositive_alpha': max(abs(r['actual_gap']) for r in records if r['alpha'] <= 0) < 1e-8,
        'negative_radial_curvature_for_positive_alpha': all(r['rho_gradient_fd'] < 0 for r in records if r['alpha'] > 0),
        'certificate_gap_upper_bound': max(r['actual_gap'] - r['certificate_bound'] for r in records) < 1e-8,
        'equality_when_z_feasible_positive_alpha': max(r['equality_abs_error'] for r in records if r['expected_equality']) < 1e-8,
    }
    report = {
        'status': 'PASS' if all(tests.values()) else 'FAIL',
        'scope': '84 deterministic finite-case checks, not a proof or learned-head experiment',
        'formula': 'z=s-alpha*W^-1*(-1,e); rho=-I*alpha; gap<=n^T W^-1 n/(2 I^2)*max(-rho,0)^2',
        'raw_intensity_map': 'I=exp(a), strictly positive and regular at I=1.2',
        'comparison': 'independent convex Stokes-coordinate QP via SciPy SLSQP',
        'software': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'finite_difference': {'gradient_step': fd_gradient_step, 'curvature_step': fd_curvature_step},
        'solver': {'method': 'SLSQP', 'ftol': 1e-13, 'maxiter': 1000, 'initial_Stokes': [1.5,0,0]},
        'matrices': matrices, 'tests': tests,
        'summary': {
            'cases': len(records),
            'nonpositive_alpha_global_cases': sum(r['alpha'] <= 0 for r in records),
            'positive_alpha_strict_saddle_cases': sum(r['alpha'] > 0 for r in records),
            'positive_alpha_feasible_z_equality_cases': sum(r['expected_equality'] for r in records),
            'positive_alpha_infeasible_z_strict_bound_cases': sum(r['alpha'] > 0 and not r['z_feasible'] for r in records),
            'max_full_raw_gradient_inf': max(r['full_raw_gradient_inf'] for r in records),
            'max_raw_gradient_fd_error_inf': max(r['raw_gradient_fd_error_inf'] for r in records),
            'max_rho_gradient_fd_error': max(r['rho_gradient_fd_error'] for r in records),
            'max_rho_energy_fd_error': max(r['rho_energy_fd_error'] for r in records),
            'max_solver_cone_violation': max(r['qp']['cone_violation'] for r in records),
            'max_solver_stationarity_inf': max(r['qp']['stationarity_inf'] for r in records),
            'max_solver_complementarity_abs': max(r['qp']['complementarity_abs'] for r in records),
            'max_certificate_violation': max(r['actual_gap']-r['certificate_bound'] for r in records),
            'max_equality_abs_error': max(r['equality_abs_error'] for r in records if r['expected_equality']),
            'max_nonpositive_alpha_gap_abs': max(abs(r['actual_gap']) for r in records if r['alpha'] <= 0),
            'solver_status_counts': {str(status): sum(r['qp']['status'] == status for r in records) for status in sorted({r['qp']['status'] for r in records})},
        },
        'records': records,
    }
    def serialize(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (HERE / 'results' / 'general_metric_independent.json').write_text(json.dumps(report, indent=2, default=serialize) + '\n')
    print(json.dumps({k:report[k] for k in ['status','tests','summary']}, indent=2, default=serialize))
    if not all(tests.values()):
        raise SystemExit('Finite-case validation failed; inspect retained records before using the result.')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig, ax = plt.subplots(figsize=(3.55, 3.0), layout='constrained')
    colors = ['#555555','#0072B2','#D55E00','#009E73']
    for matrix, color in zip(matrices, colors):
        rr = [r for r in records if r['matrix'] == matrix['name'] and r['alpha'] > 0]
        ax.scatter([r['certificate_bound'] for r in rr], [r['actual_gap'] for r in rr],
                   s=18, edgecolor=color, facecolor='none', linewidth=.8,
                   label=matrix['name'].replace('_',' '))
    upper=max(r['certificate_bound'] for r in records)*1.03
    ax.plot([0,upper],[0,upper],':',color='.25',lw=.9,label='Certificate equality')
    ax.set(xlabel='Curvature certificate upper bound', ylabel='QP-verified objective gap',
           xlim=(-.1, upper), ylim=(-.1, upper))
    ax.grid(alpha=.15); ax.legend(fontsize=6.7,loc='upper left')
    fig.savefig(HERE.parent/'latex'/'images'/'general_metric_certificate.pdf')
    fig.savefig(HERE.parent/'latex'/'images'/'general_metric_certificate.png',dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()
