"""Frozen follow-up: coordinate repair and explicit physical-space controls.

This is development on previously inspected data, not an independent test set.
No training, network requests, outcome-selected inputs, or hidden gradients.
"""
from pathlib import Path
import argparse, collections, datetime, hashlib, importlib.util, json, platform, time
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('core', ROOT/'verification/verify_warm_start.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
TOL = 1e-6
ARMS = ['warm_start', 'cold_start', 'canonical_warm', 'audited_cold', 'audited_chart', 'guarded_chart']
METHODS = ['BFGS', 'L-BFGS-B']


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, obj): p.write_text(json.dumps(c.pack(obj), indent=2, allow_nan=False)+'\n')


def sources():
    return {'real_inputs': ROOT/'application/real_data/INPUTS.json',
            'real_results': ROOT/'application/real_data/RESULTS.json',
            'fresnel_inputs': ROOT/'application/fresnel_sequence/inputs.json',
            'core': ROOT/'verification/verify_warm_start.py', 'algorithm': Path(__file__)}


def protocol():
    return dict(registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        design_status='Retrospective method development after inspecting the original real-data audit. Freeze this follow-up before new method outcomes. Fresnel is a previously used mechanism control, not held-out validation.',
        source_sha256={k: sha(p) for k,p in sources().items()},
        observations='All 4642 existing real-data entries and all 984 existing Fresnel frames, with no new selection, reordering, noise, normalization, thresholds, seeds or data. Preserve original strata and dependencies.',
        objective='Unchanged nominal four-analyzer unweighted quadratic, W=diag(1,0.5,0.5).',
        methods=METHODS, arms=ARMS+['physical_projected_gradient','direct_cone'],
        gradient_tolerance=TOL, objective_gap_threshold=TOL,
        solver_options='Unchanged core: gtol=1e-6, maxiter=500; LBFGSB ftol=0, maxfun=5000, maxls=40. Each new minimize invocation resets quasi-Newton memory.',
        initialization='Real first/cold: [log(max(S0_LS,1e-6)),.2,-.1]; Fresnel first/cold: [0,.2,-.1]. Each arm carries its own endpoint, including rejected states; no oracle solution initializes iterative arms.',
        canonical_warm='Before each solve replace a previous outer w by w/(w dot w), preserving its physical output. No stopping audit in this arm.',
        audited_cold='Raw warm solve, physical bound audit; at most one cold-start solve if rejected, then audit again.',
        audited_chart='Raw warm solve and physical bound. On rejection, if abs(r-1)<=1e-3 query one full analytic head Hessian; if c_r/I^2 < -1e-7 choose the lowest objective among radial offsets -.01,0,+.01, canonicalize this candidate, and solve once. Otherwise an outer endpoint is canonicalized and solved once. Inner nonfold endpoints are left rejected. At most one such repair.',
        guarded_chart='Same audited chart; if still rejected, use the standard physical projected-gradient control from the current physical state. This fallback, not the coordinate change, is responsible for any additional repaired outputs.',
        physical_projected_gradient='Euclidean cone projection after s-g, step 1 since lambda_max(W)=1; audit before and after updates; at most 64 updates. Terminate only if the existing physical bound <=1e-6. Native head flags do not accept physical outputs.',
        physical_lift='Map a corrected positive-intensity physical point to the inner inverse. An exact apex has no finite exponential lift, is returned in physical coordinates and makes the next head start cold. Clamp DoLP only to [0,1] to remove cone-projection roundoff when lifting; log physical-lift discrepancy.',
        direct_cone='Existing exact nominal cone reduction, timed as a direct baseline, with no iterative arm using its answer.',
        paired_endpoint_audit='All 9284 stored raw real-data warm endpoints are audited. Apply the cold, chart, and chart-plus-physical procedures to every bound rejection, independent of reference gap. Preserve all non-rejections in denominator.',
        metrics='Reference gap, distance to direct cone solution, native and monitored acceptance separately, unresolved bound rejections, native FG pairs, physical-gradient audits/updates, full Hessians, trial FG pairs, coordinate switches, optimizer calls, observed wall time. Report strata and full histories.',
        timing='One sequential local CPU execution, fixed arm order; no timing superiority or significance inferred from a single pass. Count-based comparison primary. Direct controls included.',
        exclusions='No adaptive run expansion, threshold tuning, discarded failures, remote compute, training, external submission, or claim that frozen development data constitute generalization.',
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))


def cold(y, dataset):
    return np.array([0., .2, -.1]) if dataset=='Fresnel' else np.r_[np.log(max(sum(y)/2,1e-6)),.2,-.1]


def canonical(x):
    x=np.array(x,dtype=float,copy=True); rr=float(x[1:]@x[1:])
    if rr>1: x[1:]/=rr; return x,True
    return x,False


def project(s):
    t=float(s[0]); p=np.asarray(s[1:]); rho=float(np.hypot(*p))
    if rho<=t: return np.array(s,copy=True)
    if rho<=-t: return np.zeros(3)
    return np.r_[(t+rho)/2, ((t+rho)/(2*rho))*p]


def lift(s):
    I=float(s[0])
    if I<=0: return None
    q=np.asarray(s[1:])/I; d=float(np.hypot(*q)); bounded=min(1.,max(0.,d))
    w=q/(1+np.sqrt(max(0.,1-bounded*bounded)))
    return np.r_[np.log(I), w]


def counter():
    return dict(native_fg_pairs=0,trial_fg_pairs=0,physical_gradients=0,full_head_hessians=0,
                coordinate_switches=0,optimizer_calls=0,pg_updates=0,nonfinite_calls=0)


def audit(s,y,counts,with_gradient=False):
    counts['physical_gradients']+=1
    if not np.all(np.isfinite(s)): return (np.inf,np.full(3,np.nan)) if with_gradient else np.inf
    g=c.A.T@(c.A@s-y); rho=float(np.hypot(*s[1:]))
    # The supporting direction is the physical direction, unchanged by a sheet switch.
    e=s[1:]/rho if rho else np.array([1.,0.])
    bound=c.certificate(s,g,0. if rho==0 else 1.,e)
    return (bound,g) if with_gradient else bound


def native(x,y,method,counts):
    z=c.solve(x,y,method,TOL)
    counts['native_fg_pairs']+=z['counts']['function'];counts['optimizer_calls']+=1
    counts['nonfinite_calls']+=z['counts']['nonfinite_calls']
    return z


def pg(s,y,counts):
    s=np.array(s,copy=True); bounds=[]
    for k in range(65):
        bound,g=audit(s,y,counts,with_gradient=True);bounds.append(bound)
        if np.isfinite(bound) and bound<=TOL: break
        if k==64: break
        # Reuse the gradient just acquired by audit; count one physical query per step.
        s=project(s-g);counts['pg_updates']+=1
    return s,bound,bounds


def repair(base,y,method,arm,dataset,counts):
    s=c.geometry(base['x'])[0];bound=audit(s,y,counts);events=[];stage=[];last=base
    if not np.isfinite(bound) or bound>TOL:
        if arm=='audited_cold':
            last=native(cold(y,dataset),y,method,counts);events.append('cold_restart')
        else:
            x=np.array(base['x']);r=float(np.linalg.norm(x[1:]));fold=False
            if abs(r-1)<=1e-3:
                counts['full_head_hessians']+=1
                e=x[1:]/r if r else np.array([1.,0.]);v=np.r_[0.,e]
                cr=float(v@c.hessian(x,y)@v)
                fold=bool(cr/s[0]**2 < -1e-7)
            if fold:
                candidates=[x+np.r_[0.,h*x[1:]/r] for h in [-.01,0.,.01]]
                losses=[c.fg(z,y)[0] for z in candidates];counts['trial_fg_pairs']+=3
                seed,changed=canonical(candidates[int(np.argmin(losses))])
                counts['coordinate_switches']+=int(changed)
                last=native(seed,y,method,counts);events.append('fold_restart')
            elif r>1:
                seed,changed=canonical(x);counts['coordinate_switches']+=int(changed)
                last=native(seed,y,method,counts);events.append('outer_restart')
        if events:
            s=c.geometry(last['x'])[0];bound=audit(s,y,counts)
        stage.append(dict(stage='head_repair',bound=bound,events=list(events)))
    physical_fallback=False;pg_bounds=[]
    if arm=='guarded_chart' and (not np.isfinite(bound) or bound>TOL):
        s,bound,pg_bounds=pg(s,y,counts);physical_fallback=True;events.append('physical_fallback')
        x=lift(s)
    else: x=np.array(last['x'])
    return dict(x=x,physical=s,bound=bound,accepted=bool(np.isfinite(bound) and bound<=TOL),
                native_success=None if physical_fallback else last['success'],
                initial_native_success=base['success'],events=events,stages=stage,
                physical_fallback=physical_fallback,pg_bounds=pg_bounds,
                warnings=dict(collections.Counter(base['warnings'])+collections.Counter(last['warnings'])) if last is not base else base['warnings'],counts=counts)


def head_fit(previous,y,method,arm,dataset):
    counts=counter();seed=cold(y,dataset) if arm=='cold_start' or previous is None else np.array(previous)
    if arm=='canonical_warm':seed,changed=canonical(seed);counts['coordinate_switches']+=int(changed)
    base=native(seed,y,method,counts)
    if arm.startswith('audited') or arm=='guarded_chart':
        return repair(base,y,method,arm,dataset,counts)
    s=c.geometry(base['x'])[0]
    return dict(x=base['x'],physical=s,bound=None,accepted=base['success'],
                native_success=base['success'],initial_native_success=base['success'],events=[],stages=[],
                physical_fallback=False,pg_bounds=[],warnings=base['warnings'],counts=counts)


def all_streams():
    real=json.loads(sources()['real_inputs'].read_text())
    for s in real:s['observations']=np.asarray(s['observations']);s['suite']='released'
    sim=json.loads(sources()['fresnel_inputs'].read_text())
    out=list(real)
    for s in sim:
        out.append(dict(dataset='Fresnel',suite='simulation',sample=str(s['sequence']),
                        stream=f'Fresnel:{s["sequence"]}',stratum='fixed_physical_grid',
                        kind='simulated_sequence',observations=np.asarray(s['observations'])))
    return out


def measured(out,y,minimum,opt):
    s=out['physical'];res=c.A@s-y;gap=.5*float(res@res)-minimum
    extra=counter();bound=audit(s,y,extra)
    inaccurate=bool(not np.isfinite(gap) or gap>TOL)
    if np.isfinite(gap) and np.isfinite(bound):assert gap<=bound+1e-10
    if out.get('bound') is not None:assert np.isclose(out['bound'],bound,rtol=1e-9,atol=1e-14)
    return dict(gap=gap,inaccurate=inaccurate,false_accept=bool(out['accepted'] and inaccurate),
        native_false_success=bool(out['native_success'] is True and inaccurate),
        physical_bound=bound,physical_accept=bool(np.isfinite(bound) and bound<=TOL),
        distance_to_cone_solution=float(np.linalg.norm(s-opt)),
        physical=s,x=out['x'],native_success=out['native_success'],accepted=out['accepted'],
        counts=out['counts'],events=out['events'],warnings=out.get('warnings',{}),
        pg_bounds=out.get('pg_bounds',[]),stages=out.get('stages',[]))


def summaries(rows,fields=('suite','dataset','stratum','method','arm')):
    groups=collections.defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in fields)].append(r)
    out=[]
    for keys,rs in groups.items():
        d=dict(zip(fields,keys));d.update(entries=len(rs),
            inaccurate=sum(r['inaccurate'] for r in rs),false_accept=sum(r['false_accept'] for r in rs),
            native_false_success=sum(r['native_false_success'] for r in rs),
            native_nonaccept=sum(r['native_success'] is False for r in rs),
            monitored_rejections=sum(not r['physical_accept'] for r in rs),
            accepted=sum(r['accepted'] for r in rs),
            max_gap=max(r['gap'] for r in rs),max_distance=max(r['distance_to_cone_solution'] for r in rs),
            elapsed_seconds=sum(r['elapsed_seconds'] for r in rs),
            events=dict(collections.Counter(e for r in rs for e in r['events'])),
            counts={k:sum(r['counts'][k] for r in rs) for k in counter()})
        out.append(d)
    return out


def preflight():
    rows=[];max_value=max_gradient=max_lift=0.
    for I in [.005,1.,4.]:
        for r in [0.,1e-7,.5,.9999,1.,1.0001,2.,100.,1e7]:
            x=np.r_[np.log(I),r*np.array([.6,.8])];z,changed=canonical(x)
            sx,Dx=c.geometry(x);sz,Dz=c.geometry(z)
            value=float(np.max(np.abs(sx-sz)));max_value=max(max_value,value)
            g=np.array([.3,-.4,.7]);jx=Dx.T@g;jz=Dz.T@g
            if changed:
                H=np.eye(2)-2*np.outer(x[1:]/r,x[1:]/r)
                error=float(np.max(np.abs(jz[1:]-r*r*H@jx[1:])))
                max_gradient=max(max_gradient,error)
            lx=lift(sz);max_lift=max(max_lift,float(np.max(np.abs(c.geometry(lx)[0]-sz))))
            rows.append(dict(I=I,r=r,changed=changed,physical_difference=value))
    assert max_value<1e-14 and max_gradient<1e-13 and max_lift<1e-14
    for s in [[-1.,0.,0.],[0.,2.,0.],[1.,.2,.3],[1.,1.,0.]]:
        p=project(np.array(s));assert p[0]>=np.linalg.norm(p[1:])-1e-14
    return dict(states=rows,max_output_difference=max_value,
                max_gradient_transform_difference=max_gradient,max_lift_difference=max_lift)


def paired(outdir):
    historical=json.loads(sources()['real_results'].read_text())['records']
    obs={s['stream']:s['observations'] for s in all_streams() if s['suite']=='released'}
    rows=[]
    for r in historical:
        if r['arm']!='warm_start':continue
        y=obs[r['stream']][r['index']];opt,minimum=c.cone_solution(y)
        for arm in ['audited_cold','audited_chart','guarded_chart']:
            counts=counter();base=dict(x=np.array(r['final_x']),success=r['success'],warnings={})
            start=time.perf_counter();result=repair(base,y,r['method'],arm,r['dataset'],counts)
            elapsed=time.perf_counter()-start
            row=measured(result,y,minimum,opt)
            row.update(suite='paired',dataset=r['dataset'],stratum=r['stratum'],method=r['method'],arm=arm,
                       stream=r['stream'],index=r['index'],elapsed_seconds=elapsed,
                       original_gap=r['gap'],original_false_success=r['false_success'])
            rows.append(row)
    write(outdir/'PAIRED.json',dict(records=rows,summary=summaries(rows)))
    return summaries(rows,('suite','method','arm'))


def run(outdir, replay=False):
    registered=json.loads((outdir/'PROTOCOL.json').read_text())
    for k,p in sources().items():assert sha(p)==registered['source_sha256'][k],k
    assert not (outdir/'RESULTS.json').exists(),'Preserve the frozen outcome.'
    write(outdir/'PREFLIGHT.json',preflight())
    streams=all_streams();rows=[];started=time.time()
    history=json.loads(sources()['real_results'].read_text())['records']
    ref={(r['stream'],r['index'],r['method'],r['arm']):r for r in history if r['arm'] in ['warm_start','cold_start']}
    compared=0;max_baseline_difference=0.
    for si,stream in enumerate(streams):
        dataset=stream['dataset'];states={(m,a):cold(stream['observations'][0],dataset) for m in METHODS for a in ARMS}
        physical_state=c.geometry(cold(stream['observations'][0],dataset))[0]
        for index,y in enumerate(stream['observations']):
            opt,minimum=c.cone_solution(y)
            for (method,arm),previous in states.items():
                start=time.perf_counter();out=head_fit(previous,y,method,arm,dataset);elapsed=time.perf_counter()-start
                row=measured(out,y,minimum,opt)
                row.update(suite=stream['suite'],dataset=dataset,stratum=stream['stratum'],stream=stream['stream'],
                           sample=stream['sample'],index=index,method=method,arm=arm,elapsed_seconds=elapsed)
                rows.append(row);states[(method,arm)]=out['x']
                if dataset!='Fresnel' and arm in ['warm_start','cold_start']:
                    old=ref[(stream['stream'],index,method,arm)];compared+=1
                    error=float(np.max(np.abs(np.array(old['final_x'])-out['x'])))
                    max_baseline_difference=max(max_baseline_difference,error)
                    assert error<1e-10 and row['native_false_success']==old['false_success']
            counts=counter();start=time.perf_counter();s,bound,bs=pg(physical_state,y,counts);elapsed=time.perf_counter()-start
            physical_state=s
            out=dict(x=lift(s),physical=s,bound=bound,accepted=bool(np.isfinite(bound) and bound<=TOL),
                     native_success=None,counts=counts,events=[],pg_bounds=bs)
            row=measured(out,y,minimum,opt);row.update(suite=stream['suite'],dataset=dataset,
                stratum=stream['stratum'],stream=stream['stream'],sample=stream['sample'],index=index,
                method='physical',arm='physical_projected_gradient',elapsed_seconds=elapsed);rows.append(row)
            start=time.perf_counter();s,_=c.cone_solution(y);elapsed=time.perf_counter()-start
            out=dict(x=None,physical=s,bound=None,accepted=True,native_success=None,counts=counter(),events=[])
            row=measured(out,y,minimum,opt);row.update(suite=stream['suite'],dataset=dataset,
                stratum=stream['stratum'],stream=stream['stream'],sample=stream['sample'],index=index,
                method='physical',arm='direct_cone',elapsed_seconds=elapsed);rows.append(row)
        if si%30==0:print(json.dumps(dict(streams_completed=si+1,total=len(streams),seconds=round(time.time()-started,1))),flush=True)
    summary=dict(protocol_sha256=sha(outdir/'PROTOCOL.json'),records=len(rows),
        original_real_baseline_endpoints_compared=compared,max_baseline_coordinate_difference=max_baseline_difference,
        elapsed_seconds=time.time()-started,summary=summaries(rows),
        aggregate=summaries(rows,('suite','method','arm')),
        paired_summary=paired(outdir),frozen_development_data=True)
    write(outdir/'RESULTS.json',dict(records=rows,summary=summary['summary']));write(outdir/'SUMMARY.json',summary)
    print(json.dumps(c.pack(summary['aggregate'])),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--register',action='store_true');ap.add_argument('--replay-from',type=Path)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.register:
        assert not any(out.iterdir()),'Choose a new empty run directory.';write(out/'PROTOCOL.json',protocol());print('Protocol frozen; outcomes not evaluated.');return
    if args.replay_from:
        assert not any(out.iterdir());src=args.replay_from.resolve()
        (out/'PROTOCOL.json').write_bytes((src/'PROTOCOL.json').read_bytes())
    run(out)
    if args.replay_from:
        expected=json.loads((src/'SUMMARY.json').read_text());actual=json.loads((out/'SUMMARY.json').read_text())
        def scientific(x):
            if isinstance(x,dict):return {k:scientific(v) for k,v in x.items() if k not in ['elapsed_seconds','protocol_sha256']}
            if isinstance(x,list):return [scientific(v) for v in x]
            return x
        assert scientific(expected)==scientific(actual)
        write(out/'REPRODUCTION_CHECK.json',dict(scientific_summary_exact_match=True,timing_excluded=True))


if __name__=='__main__':main()
