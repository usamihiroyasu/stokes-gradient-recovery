"""Fold gradient recovery and dual objective-gap bound, deterministic cases.

Run beside calibrated_certificate.py. Uses relative sibling imports and emits
its JSON beside this file. No rendering, fitting retained data, or neural training.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
BASE = HERE / 'calibrated_certificate.py'
spec = importlib.util.spec_from_file_location('independent_cone_qp', BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def value_raw_gradient(intensity, v, z, w):
    """Coordinates are (I,w), not (log I,w): b = partial L / partial I."""
    den = 1 + v @ v
    f = 2 * v / den
    df = 2 * np.eye(2) / den - 4 * np.outer(v, v) / den**2
    s = intensity * np.r_[1., f]
    g = w @ (s - z)
    b = float(np.r_[1., f] @ g)
    gw = intensity * df.T @ g[1:]
    return float(.5 * (s-z) @ g), b, gw


def recover_gradient(b, gw, gamma, intensity, e):
    tangent = np.array([-e[1], e[0]])
    return np.r_[b + gamma/intensity,
                 -gamma/intensity*e + float(tangent @ gw)/intensity*tangent]


def certificate(g, w, normal):
    winv_g = np.linalg.solve(w, g)
    winv_n = np.linalg.solve(w, normal)
    denominator = float(normal @ winv_n)
    multiplier = max(0., -float(normal @ winv_g)/denominator)
    residual = g + multiplier*normal
    return float(.5 * residual @ np.linalg.solve(w, residual)), multiplier


def main():
    specs = [
        ('ideal_uncorrelated', [0,45,90,135], [1]*4, [1]*4, [1]*4),
        ('nonideal_angles', [0,37,91,143], [1.03,.97,1.01,.99], [.94]*4, [1,.8,1.1,1.3]),
        ('gain_and_efficiency', [3,44,86,132], [1.15,.88,1.04,.94], [.91,.95,.89,.97], [1.2,.7,1.4,.9]),
        ('heteroscedastic', [-5,40,95,139], [.93,1.08,.96,1.02], [.90,.96,.93,.88], [.6,1.7,.85,1.2]),
    ]
    zchoices = [
        ('interior_0', 'interior', [.8,0.,0.]),
        ('interior_1', 'interior', [1.7,.2,-.1]),
        ('interior_2', 'interior', [.45,-.12,.18]),
        ('boundary_0', 'boundary', [.9,.9,0.]),
        ('boundary_1', 'boundary', [1.4,.6,float(np.sqrt(1.4**2-.6**2))]),
        ('boundary_2', 'boundary', [1.,-.6,.8]),
        ('outside_0', 'outside', [.2,1.1,-.4]),
        ('outside_1', 'outside', [.6,-1.3,.2]),
        ('outside_2', 'outside', [.05,.5,.9]),
    ]
    intensity, step = 1.2, 1e-4
    rows, matrices = [], []
    for name, angles, gain, efficiency, variance in specs:
        a, effective_variance, w = base.analyzer_metric(angles,gain,efficiency,variance)
        matrices.append({'name':name,'A':a.tolist(), 'W':w.tolist(),
                         'effective_observation_variance':effective_variance.tolist()})
        for angle in [0., .7, 1.9]:
            e = np.array([np.cos(angle),np.sin(angle)])
            normal = np.r_[-1.,e]
            s = intensity*np.r_[1.,e]
            for zname, category, zlist in zchoices:
                z=np.array(zlist)
                value,b,gw=value_raw_gradient(intensity,e,z,w)
                gs=w@(s-z)
                gamma_exact=-intensity*float(e@gs[1:])
                gwplus=value_raw_gradient(intensity,e+step*e,z,w)[2]
                gwminus=value_raw_gradient(intensity,e-step*e,z,w)[2]
                vector_hvp=(gwplus-gwminus)/(2*step)
                gamma_fd=float(e@vector_hvp)
                gamma_energy=(value_raw_gradient(intensity,e+step*e,z,w)[0]-2*value
                              +value_raw_gradient(intensity,e-step*e,z,w)[0])/step**2
                recovered_exact=recover_gradient(b,gw,gamma_exact,intensity,e)
                recovered_fd=recover_gradient(b,gw,gamma_fd,intensity,e)
                recovered_vector_hvp=np.r_[b+float(e@vector_hvp)/intensity,-vector_hvp/intensity]
                bound_exact,lam_exact=certificate(recovered_exact,w,normal)
                bound_fd,lam_fd=certificate(recovered_fd,w,normal)
                bound_vector,lam_vector=certificate(recovered_vector_hvp,w,normal)
                _,qp=base.solve_qp(z,w)
                gap=value-qp['objective']
                rows.append({
                    'matrix':name,'angle_radians':angle,'z_name':zname,'z_category':category,
                    'z':zlist,'z_cone_slack':base.slack(z),'I':intensity,'e':e.tolist(),
                    'b_partial_I':b,'g_w':gw.tolist(),'raw_gradient_norm':float(np.linalg.norm(np.r_[b,gw])),
                    'true_Stokes_gradient':gs.tolist(),'gamma_exact':gamma_exact,
                    'gamma_fd_hvp':gamma_fd,'gamma_fd_energy':float(gamma_energy),
                    'gamma_hvp_abs_error':abs(gamma_fd-gamma_exact),
                    'vector_HVP_fd':vector_hvp.tolist(),
                    'vector_HVP_identity_error_inf':float(np.max(abs(vector_hvp+intensity*gs[1:]))),
                    'gamma_energy_abs_error':float(abs(gamma_energy-gamma_exact)),
                    'recovered_gradient_exact':recovered_exact.tolist(),
                    'recovered_gradient_fd':recovered_fd.tolist(),
                    'recovery_exact_error_inf':float(np.max(abs(recovered_exact-gs))),
                    'recovery_fd_error_inf':float(np.max(abs(recovered_fd-gs))),
                    'recovered_gradient_vector_HVP':recovered_vector_hvp.tolist(),
                    'recovery_vector_HVP_error_inf':float(np.max(abs(recovered_vector_hvp-gs))),
                    'lambda_star_exact':lam_exact,'lambda_star_fd':lam_fd,
                    'lambda_star_vector_HVP':lam_vector,
                    'certificate_bound_exact':bound_exact,'certificate_bound_fd':bound_fd,
                    'certificate_bound_vector_HVP':bound_vector,
                    'fold_objective':value,'qp':qp,'actual_gap':gap,
                    'violation_exact':gap-bound_exact,'violation_fd':gap-bound_fd,
                    'violation_vector_HVP':gap-bound_vector,
                })
    tests={
        'all_108_solver_success':len(rows)==108 and all(r['qp']['success'] for r in rows),
        'solver_KKT_stationarity':max(r['qp']['stationarity_inf'] for r in rows)<2e-6,
        'solver_cone_feasible':max(r['qp']['cone_violation'] for r in rows)<1e-8,
        'solver_complementarity':max(r['qp']['complementarity_abs'] for r in rows)<1e-8,
        'all_raw_gradients_nonzero':min(r['raw_gradient_norm'] for r in rows)>1e-6,
        'exact_gradient_recovery':max(r['recovery_exact_error_inf'] for r in rows)<1e-12,
        'HVP_gradient_recovery':max(r['recovery_fd_error_inf'] for r in rows)<1e-6,
        'vector_HVP_identity':max(r['vector_HVP_identity_error_inf'] for r in rows)<1e-6,
        'vector_HVP_full_gradient_recovery':max(r['recovery_vector_HVP_error_inf'] for r in rows)<1e-6,
        'independent_HVP_gamma':max(r['gamma_hvp_abs_error'] for r in rows)<1e-6,
        'independent_energy_gamma':max(r['gamma_energy_abs_error'] for r in rows)<2e-6,
        'exact_certificate_gap':max(r['violation_exact'] for r in rows)<1e-8,
        'HVP_certificate_gap':max(r['violation_fd'] for r in rows)<1e-7,
        'vector_HVP_certificate_gap':max(r['violation_vector_HVP'] for r in rows)<1e-7,
    }
    summary={
        'cases':len(rows),'categories':{c:sum(r['z_category']==c for r in rows) for c in ['interior','boundary','outside']},
        'solver_status_counts':{str(k):sum(r['qp']['status']==k for r in rows) for k in sorted({r['qp']['status'] for r in rows})},
        'min_raw_gradient_norm':min(r['raw_gradient_norm'] for r in rows),
        'max_gradient_recovery_exact_error_inf':max(r['recovery_exact_error_inf'] for r in rows),
        'max_gradient_recovery_HVP_error_inf':max(r['recovery_fd_error_inf'] for r in rows),
        'max_vector_HVP_identity_error_inf':max(r['vector_HVP_identity_error_inf'] for r in rows),
        'max_gradient_recovery_vector_HVP_error_inf':max(r['recovery_vector_HVP_error_inf'] for r in rows),
        'max_gamma_HVP_abs_error':max(r['gamma_hvp_abs_error'] for r in rows),
        'max_gamma_energy_abs_error':max(r['gamma_energy_abs_error'] for r in rows),
        'max_solver_KKT_stationarity':max(r['qp']['stationarity_inf'] for r in rows),
        'max_solver_cone_violation':max(r['qp']['cone_violation'] for r in rows),
        'max_solver_complementarity':max(r['qp']['complementarity_abs'] for r in rows),
        'max_certificate_violation_exact':max(r['violation_exact'] for r in rows),
        'max_certificate_violation_HVP':max(r['violation_fd'] for r in rows),
        'max_certificate_violation_vector_HVP':max(r['violation_vector_HVP'] for r in rows),
        'certificate_multiplier_zero_cases':sum(r['lambda_star_exact']==0 for r in rows),
        'certificate_multiplier_positive_cases':sum(r['lambda_star_exact']>0 for r in rows),
        'actual_gap_range':[min(r['actual_gap'] for r in rows),max(r['actual_gap'] for r in rows)],
    }
    report={
        'status':'PASS' if all(tests.values()) else 'FAIL',
        'scope':'108 deterministic arbitrary-z fold cases; finite checks, not a proof or trained-head evaluation',
        'coordinate_convention':'b=partial L/partial I in raw (I,w), not log-intensity raw derivative',
        'formulae':{
            'gamma':'-I e dot gp; also radial raw Hessian curvature',
            'g_recovered':'(b+gamma/I, -gamma e/I + (eperp dot gw/I) eperp)',
            'vector_HVP_identity':'v=H_ww e=-I*gp',
            'vector_HVP_recovery':'(b+(e dot v)/I, -v/I)',
            'bound':'.5 min_lambda>=0 (g_recovered+lambda*n)^T W^-1 (g_recovered+lambda*n)',
            'lambda_star':'max(0,-n^T W^-1 g_recovered / (n^T W^-1 n))',
        },
        'gamma_HVP_method':'central difference of independently evaluated raw g_w along e, step 1e-4',
        'solver':'same independent Stokes-coordinate SciPy SLSQP convex cone QP as sibling script',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'solver_source_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
        'software':{'numpy':np.__version__,'scipy':scipy.__version__},
        'matrices':matrices,'tests':tests,'summary':summary,'records':rows,
    }
    def serialize(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    (HERE/'results'/'fold_gradient_recovery_independent.json').write_text(json.dumps(report,indent=2,default=serialize)+'\n')
    print(json.dumps({k:report[k] for k in ['status','tests','summary']},indent=2,default=serialize))
    if not all(tests.values()):raise SystemExit('Verification failed; inspect retained records.')


if __name__=='__main__':main()
