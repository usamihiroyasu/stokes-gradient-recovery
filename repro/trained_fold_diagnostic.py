"""Measure fold-candidate incidence after fitting analyzer observations."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import torch

from trained_fold_core import train_case

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
W = np.array([1., .5, .5])
M = 1 / W
THRESHOLDS = [1e-3, 1e-4, 1e-5]


def diagnose(intensity, raw, observations, analyzer):
    I = np.asarray(intensity, dtype=float)
    w = np.asarray(raw, dtype=float)
    y, A = np.asarray(observations, float), np.asarray(analyzer, float)
    r = np.linalg.norm(w, axis=1)
    e = np.divide(w, r[:, None], out=np.tile([1., 0.], (len(r), 1)), where=r[:, None] > 0)
    phi = 2 * r / (1 + r * r)
    ar = 2 * (1 - r * r) / (1 + r * r)**2
    tr = 2 / (1 + r * r)
    kr = 4 * r * (r * r - 3) / (1 + r * r)**3
    s = I[:, None] * np.column_stack((np.ones(len(r)), phi[:, None] * e))
    residual = s @ A.T - y
    loss = .5 * (residual * residual).sum(1)
    g = residual @ A
    radial_g = (g[:, 1:] * e).sum(1)
    j = I[:, None] * (ar[:, None] * radial_g[:, None] * e
                      + tr[:, None] * (g[:, 1:] - radial_g[:, None] * e))
    b = g[:, 0] + phi * radial_g
    intensity_prime = -np.expm1(-(I - 1e-6))
    stationarity = np.sqrt((intensity_prime * b)**2 + (j*j).sum(1)) / np.maximum(I**2, 1e-12)
    curvature = I**2 * ar**2 * .5 + I * kr * radial_g
    normal = np.column_stack((-np.ones(len(r)), e))
    slack = I * (1 - phi)
    multiplier = np.maximum(0., -((normal * M * g).sum(1) + slack) / 3)
    bound = .5 * ((g + multiplier[:, None] * normal)**2 * M).sum(1) + multiplier * slack
    # Exact weighted-cone minimizer for W=diag(1,1/2,1/2).
    z = (y @ A) * M
    zp = np.linalg.norm(z[:, 1:], axis=1)
    cone = z.copy()
    outside = zp > z[:, 0]
    boundary_I = np.maximum(0., (2*z[:, 0] + zp)/3)
    cone[outside, 0] = boundary_I[outside]
    cone[outside, 1:] = z[outside, 1:] * np.divide(
        boundary_I[outside], zp[outside], out=np.zeros(outside.sum()), where=zp[outside] > 0
    )[:, None]
    minimum_loss = .5*((cone @ A.T-y)**2).sum(1)
    gap = loss-minimum_loss
    assert np.min(bound-gap) > -2e-12
    best_drop = np.zeros(len(r))
    for size in [.001, .01, .05, .1]:
        for sign in [-1, 1]:
            rp = r + sign*size
            pp = 2*rp/(1+rp*rp)
            sp = I[:, None]*np.column_stack((np.ones(len(r)), pp[:, None]*e))
            value = .5*((sp@A.T-y)**2).sum(1)
            best_drop = np.maximum(best_drop, loss-value)
    assert np.max(best_drop-bound) < 2e-12
    counts = {}
    for threshold in THRESHOLDS:
        near = np.abs(r-1) <= threshold
        negative = curvature/I**2 < -1e-7
        candidate = near & negative
        stationary = candidate & (stationarity <= 1e-5)
        counts[str(threshold)] = dict(near=int(near.sum()), negative=int(candidate.sum()),
            stationary_negative=int(stationary.sum()), negative_fraction=float(candidate.mean()),
            max_trial_drop=float(best_drop[candidate].max()) if candidate.any() else 0.,
            max_drop_bound_ratio=float((best_drop[candidate]/np.maximum(bound[candidate],1e-300)).max()) if candidate.any() else 0.)
    return dict(counts=counts, mean_loss=float(loss.mean()),
        minimum_bound_margin=float((bound-gap).min()),
        minimum_radius_distance=float(np.abs(r-1).min()),
        maximum_stationarity=float(stationarity.max()),
        arrays=dict(radius=r, radial_curvature=curvature, normalized_curvature=curvature/I**2,
                    head_stationarity=stationarity, physical_gradient=g, data_loss=loss,
                    gap=gap, halfspace_bound=bound, radial_trial_drop=best_drop))


def positive_control():
    z = np.array([1., .6, .3])
    e = z[1:]/np.linalg.norm(z[1:])
    I = (2*z[0]+np.linalg.norm(z[1:]))/3
    s = I*np.r_[1., e]
    g = W*(s-z)
    gamma = -I*(e@g[1:])
    bound = 3*gamma**2/(2*I**2)
    gap = .5*((s-z)**2*W).sum()
    assert gamma < 0 and np.isclose(bound,gap,rtol=1e-12,atol=1e-14)
    # Float64 isolates the strict stationary control from float32 rounding.
    paths = {}
    for label, offset in [('stationary',0.),('perturbed',.01)]:
        a = torch.tensor(np.log(np.expm1(I)), dtype=torch.float64, requires_grad=True)
        w = torch.tensor((1+offset)*e, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([a,w],lr=.003)
        values = []
        for step in range(301):
            intensity = torch.nn.functional.softplus(a)
            state = intensity*torch.cat((torch.ones(1,dtype=w.dtype),2*w/(1+(w*w).sum())))
            objective = .5*((state-torch.tensor(z))**2*torch.tensor(W)).sum()
            values.append(float(objective.detach()))
            if step < 300:
                opt.zero_grad();objective.backward();opt.step()
        paths[label] = values
    return dict(I=I,e=e.tolist(),gradient=g.tolist(),gamma=gamma,bound=bound,gap=gap,
                loss_paths=paths,precision='float64',purpose='constructed control, excluded from incidence')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = json.loads((HERE/'trained_fold_protocol.json').read_text())
    folder = RESULTS/'trained_fold'
    folder.mkdir(exist_ok=True)
    rows = []
    started = time.monotonic()
    for dataset in protocol['datasets']:
        for initialization in protocol['initializations']:
            for seed in protocol['seeds']:
                stem = f'{dataset}_{initialization}_{seed}'
                file = folder/(stem+'.npz')
                if args.check_only:
                    stored = dict(np.load(file,allow_pickle=False))
                    checkpoints = json.loads(str(stored.pop('metrics_json')))
                else:
                    if time.monotonic()-started > 1200:
                        raise RuntimeError('Fixed 20-minute compute limit reached')
                    result = train_case(dataset,seed,initialization,HERE/'data/renderer_reference.npz')
                    stored = {k:result[k] for k in ['target','y','A','s','I','w']}
                    stored['loss_curve'] = np.asarray(result['loss_curve'])
                    stored.update({'weight__'+k:v for k,v in result['state_dict'].items()})
                    checkpoints = {}
                    for step, metrics in result['checkpoints'].items():
                        checkpoints[str(step)] = {k:v for k,v in metrics.items() if not k.endswith('_copy')}
                        for key in ['s','I','w']:
                            stored[f'checkpoint_{step}_{key}'] = metrics[key+'_copy']
                    np.savez_compressed(file,**stored,metrics_json=np.array(json.dumps(checkpoints)))
                diagnostics = {}
                for step in checkpoints:
                    report = diagnose(stored[f'checkpoint_{step}_I'],stored[f'checkpoint_{step}_w'],stored['y'],stored['A'])
                    report.pop('arrays')
                    diagnostics[step] = report
                final = diagnose(stored['I'],stored['w'],stored['y'],stored['A'])
                arrays = final.pop('arrays')
                if not args.check_only:
                    np.savez_compressed(folder/(stem+'_diagnostic.npz'),**arrays)
                rows.append(dict(dataset=dataset,initialization=initialization,seed=seed,
                    checkpoints=checkpoints,diagnostics=diagnostics,final=final,
                    record_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                print(stem,final['counts']['0.001'],flush=True)
    report = dict(protocol=protocol,environment=dict(torch=torch.__version__,numpy=np.__version__,
        python=platform.python_version(),device='cpu',threads=1),runs=rows,
        positive_control=positive_control(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),HERE/'trained_fold_core.py',HERE/'trained_fold_protocol.json']})
    path = RESULTS/'trained_fold_diagnostic.json'
    if args.check_only:
        assert report == json.loads(path.read_text())
        print('Saved records and diagnostics agree')
    else:
        path.write_text(json.dumps(report,indent=2)+'\n')
        print('Completed',len(rows),'runs in',round(time.monotonic()-started,1),'seconds')


if __name__ == '__main__':
    main()
