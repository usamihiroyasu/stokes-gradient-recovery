"""Frozen retrospective validation of outward-arithmetic stopping.

No optimizer, native stopping tolerance, observations or saved states change.
Use --output NEW_DIRECTORY for reproduction without changing the frozen run.
"""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import argparse,hashlib,json,platform,time
from monitor import evaluate,reference_gap

P=Path(__file__).resolve().parent
R=P.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def stress_cases():
    rows=[]
    # Binary-exact boundary perturbations, lower/upper chart and intensity limits.
    radii=[0.,.25,.5,.75,1-2**-20,1-2**-40,1.,1+2**-40,1+2**-20,2.,1e6]
    for radius in radii:
        for a in [0.,-40.,-64.]:
            rows.append(dict(x=[a,radius,0.],y=[.75,.5,.25,.5],family='chart_and_intensity'))
    for alpha in [0.,2**-30,2**-20,2**-10,.25,-.25]:
        # A z where z=(1+alpha,1-2*alpha,0), so grad at the fold=alpha*(-1,1,0).
        rows.append(dict(x=[0.,1.,0.],y=[1-alpha/2,(1+alpha)/2,3*alpha/2,(1+alpha)/2],family='stationary_fold'))
    for x in [[-65.,0.,0.],[65.,0.,0.],[0.,float('inf'),0.],[float('nan'),0.,0.]]:
        rows.append(dict(x=x,y=[.5]*4,family='unsupported_input'))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    protocol=json.loads((P/'results/PROTOCOL.json').read_text())
    for name,h in protocol['source_sha256'].items():assert sha(R/name)==h,name
    for name,h in protocol['implementation_sha256'].items():assert sha(P/name)==h,name
    out=args.output if args.output else P/'results'
    if args.output:out.mkdir(parents=True,exist_ok=False)
    assert not (out/'SUMMARY.json').exists(),'Preserve the original run; choose a new --output directory.'
    if args.output:save(out/'PROTOCOL.json',protocol)
    streams=json.loads((R/'application/real_data/INPUTS.json').read_text())
    observations={x['stream']:x['observations'] for x in streams}
    data=[x for x in json.loads((R/'application/real_data/RESULTS.json').read_text())['records'] if x['arm']=='warm_start']
    legacy=json.loads((R/'theory/stopping_guarantee/results/FROZEN_ENDPOINT_AUDIT.json').read_text())
    assert len(data)==len(legacy)==9284
    references=[];t0=time.perf_counter()
    for row in data:
        y=observations[row['stream']][row['index']]
        g=reference_gap(row['final_x'],y,256)
        status='accurate' if g.hi*1000000<=g.ctx.q else 'inaccurate' if g.lo*1000000>g.ctx.q else 'threshold_overlap'
        references.append(dict(status=status,lower_integer=str(g.lo),upper_integer=str(g.hi),bits=256))
    reference_seconds=time.perf_counter()-t0
    save(out/'REFERENCE_INTERVALS.json',references)
    summaries=[];allrows=[]
    for bits in protocol['precisions']:
        start=time.perf_counter();checks=[]
        for i,(row,old,ref) in enumerate(zip(data,legacy,references)):
            assert (row['stream'],row['method'],row['index'])==(old['stream'],old['method'],old['index'])
            y=observations[row['stream']][row['index']]
            result=evaluate(row['final_x'],y,bits)
            if result['certified']:
                assert ref['status']=='accurate',(bits,i,ref,result)
            if result['status']!='inconclusive':
                # A necessary comparison against an independently enclosed exact gap.
                assert int(ref['lower_integer'])*(1<<bits)<=int(result['bound_upper_integer'])*(1<<256),(bits,i)
            old_pointwise=old['bounds']['pointwise']
            item=dict(case=i,dataset=row['dataset'],stratum=row['stratum'],method=row['method'],
                reference_class=ref['status'],original_false_success=row['false_success'],
                legacy_pointwise_accept=old_pointwise is not None and old_pointwise<=1e-6,**result)
            checks.append(item)
        elapsed=time.perf_counter()-start
        summary=dict(bits=bits,endpoints=len(checks),certified=sum(x['certified'] for x in checks),
            status_counts=dict(Counter(x['status'] for x in checks)),
            accepted_inaccurate=sum(x['certified'] and x['reference_class']=='inaccurate' for x in checks),
            accurate_rejected=sum(not x['certified'] and x['reference_class']=='accurate' for x in checks),
            legacy_acceptance_disagreements=sum(x['certified']!=x['legacy_pointwise_accept'] for x in checks),
            scalar_curvature_evaluations=sum(x.get('scalar_curvature_evaluations',0) for x in checks),
            full_head_hessians=sum(x['full_head_hessians'] for x in checks),elapsed_seconds=elapsed)
        summaries.append(summary);allrows.extend(checks)
        print(json.dumps(summary),flush=True)
    save(out/'ENDPOINTS.json',allrows)
    stress=[]
    for i,row in enumerate(stress_cases()):
        for bits in protocol['precisions']:
            result=evaluate(row['x'],row['y'],bits)
            if row['family']=='unsupported_input':assert not result['certified'] and result['status']=='inconclusive'
            else:
                ref=reference_gap(row['x'],row['y'],256)
                if result['certified']:assert ref.hi*1000000<=ref.ctx.q
                if result['status']!='inconclusive':assert ref.lo*(1<<bits)<=int(result['bound_upper_integer'])*(1<<256)
            stress.append(dict(case=i,family=row['family'],**result))
    save(out/'STRESS.json',stress)
    summary=dict(reference_status_counts=dict(Counter(x['status'] for x in references)),
        historical_false_successes=sum(x['false_success'] for x in data),reference_seconds=reference_seconds,
        precisions=summaries,stress_inputs=len(stress_cases()),stress_checks=len(stress),
        implementation_sha256=protocol['implementation_sha256'],environment=dict(python=platform.python_version(),platform=platform.platform()),
        new_optimizer_calls=0,changed_observations=False,
        scope='Validated arithmetic on previously inspected nominal quadratic endpoints. Timings are descriptive one-pass CPU measurements.')
    save(out/'SUMMARY.json',summary)


if __name__=='__main__':main()
