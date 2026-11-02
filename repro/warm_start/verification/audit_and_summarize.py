"""Independent constrained-solver audit and finite/nonfinite-aware reporting.

Does not rerun or expand the registered solver grid. Raw results remain intact.
"""
import collections, csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import verify_warm_start as v
H=Path(__file__).resolve().parent

def finite(x):return x is not None and np.isfinite(x)

def run():
    main=json.loads((H/'main_results.json').read_text());precision=json.loads((H/'precision_results.json').read_text())
    unique={tuple(r['observations']):r for r in main['records']};audits=[]
    for y0,r in unique.items():
        y=np.array(y0)
        # Independent physical objective, not calling the head implementation or analytic minimizer.
        def objective(s):
            residual=np.array([(s[0]+s[1])/2,(s[0]+s[2])/2,(s[0]-s[1])/2,(s[0]-s[2])/2])-y
            return .5*float(residual@residual)
        def jac(s):
            residual=np.array([(s[0]+s[1])/2,(s[0]+s[2])/2,(s[0]-s[1])/2,(s[0]-s[2])/2])-y
            return np.array([sum(residual)/2,(residual[0]-residual[2])/2,(residual[1]-residual[3])/2])
        constraint={'type':'ineq','fun':lambda s:s[0]**2-s[1]**2-s[2]**2,'jac':lambda s:np.array([2*s[0],-2*s[1],-2*s[2]])}
        result=minimize(objective,np.array([1.1,.1,.1]),jac=jac,method='SLSQP',bounds=[(0,None),(None,None),(None,None)],constraints=[constraint],options={'ftol':1e-13,'maxiter':1000})
        audits.append(dict(case=r['case'],angle_change=r['angle_change'],success=bool(result.success),status=int(result.status),message=str(result.message),gap_error=float(result.fun-r['cone_minimum']),physical=result.x.tolist(),cone_violation=max(0.,float(np.linalg.norm(result.x[1:])-result.x[0])),nfev=int(result.nfev),njev=int(result.njev)))
    a=dict(cases=len(audits),failures=sum(not r['success'] for r in audits),max_objective_discrepancy=max(abs(r['gap_error']) for r in audits),max_cone_violation=max(r['cone_violation'] for r in audits),records=audits)
    v.save('cone_slsqp_audit.json',a)
    assert a['failures']==0 and a['max_objective_discrepancy']<1e-10 and a['max_cone_violation']<1e-10
    summaries=[];failure_records=[];warnings=collections.Counter();all_recovery_errors=[];bound_violations=[]
    groups=collections.defaultdict(list)
    for r in main['records']:groups[(r['method'],r['gtol'],r['angle_change'])].append(r)
    for (method,tol,delta),rs in groups.items():
        q=dict(method=method,gtol=tol,angle_change=delta,cases=len(rs),previous_failures=sum(not r['previous']['success'] for r in rs))
        for arm in ['baseline','always_restart','triggered_restart']:
            vals=[r[arm] for r in rs];gaps=[r['gap'] for r in vals if finite(r['gap'])]
            q[arm+'_false_success']=sum(r['false_success'] for r in vals)
            q[arm+'_failures']=sum(not r['success'] for r in vals)
            q[arm+'_bad_finite_gap']=sum(finite(r['gap']) and r['gap']>v.GAP for r in vals)
            q[arm+'_nonfinite_gap']=sum(not finite(r['gap']) for r in vals)
            q[arm+'_max_finite_gap']=max(gaps)
        q['triggers']=sum(r['diagnostic']['trigger'] for r in rs)
        q['near_fold']=sum(r['diagnostic']['near_fold'] for r in rs)
        q['trigger_detected_false_success']=sum(r['diagnostic']['trigger'] and r['baseline']['false_success'] for r in rs)
        q['accurate_success_triggered']=sum(r['diagnostic']['trigger'] and r['baseline']['success'] and finite(r['baseline']['gap']) and r['baseline']['gap']<=v.GAP for r in rs)
        q['physical_accepts']=sum(r['diagnostic']['physical_accept'] for r in rs)
        q['physical_false_accepts']=sum(r['diagnostic']['physical_accept'] and (not finite(r['baseline']['gap']) or r['baseline']['gap']>v.GAP) for r in rs)
        q['physical_reject_accurate']=sum(not r['diagnostic']['physical_accept'] and finite(r['baseline']['gap']) and r['baseline']['gap']<=v.GAP for r in rs)
        q['baseline_f']=sum(r['baseline']['counts']['function'] for r in rs)
        q['baseline_h']=sum(r['baseline']['counts']['hessian'] for r in rs)
        q['always_extra_f']=sum(r['always_restart']['counts']['function']+3 for r in rs)
        q['always_extra_g']=sum(r['always_restart']['counts']['gradient']+3 for r in rs)
        q['always_extra_h']=sum(r['always_restart']['counts']['hessian'] for r in rs)
        q['trigger_extra_f']=sum(r['always_restart']['counts']['function']+3 for r in rs if r['diagnostic']['trigger'])
        q['trigger_extra_g']=sum(r['always_restart']['counts']['gradient']+3 for r in rs if r['diagnostic']['trigger'])
        q['trigger_extra_solver_h']=sum(r['always_restart']['counts']['hessian'] for r in rs if r['diagnostic']['trigger'])
        q['trigger_extra_audit_h']=q['near_fold']
        # Both diagnostic controllers first evaluate the head/physical state at each endpoint.
        q['diagnostic_state_evaluations']=len(rs)
        for r in rs:
            d=r['diagnostic'];gap=r['baseline']['gap']
            if finite(gap) and finite(d['physical_bound']) and gap>d['physical_bound']+1e-12:bound_violations.append([method,tol,delta,r['case']])
            if d['recovered_gradient_error'] is not None:all_recovery_errors.append(d['recovered_gradient_error'])
            for arm in ['previous','baseline','always_restart']:
                val=r[arm];warnings.update(val['warnings'])
                if not val['success'] or val['counts']['nonfinite_calls']:
                    failure_records.append(dict(method=method,gtol=tol,angle_change=delta,case=r['case'],arm=arm,previous_success=r['previous']['success'],result=val,baseline_diagnostic=d))
        summaries.append(q)
    v.save('audited_summary.json',dict(summary=summaries,cone_audit={k:z for k,z in a.items() if k!='records'},recovered_near_fold_cases=len(all_recovery_errors),max_recovered_gradient_error=max(all_recovery_errors),physical_bound_violations=bound_violations,warning_counts_across_saved_rows=dict(warnings),warning_note='Previous fits reused across next_DoLP and angle-change rows; warning totals here count row references, not unique executions'))
    v.save('failure_records.json',dict(records=failure_records,nonfinite_serialization='Nonfinite numeric values are null in raw JSON, with status/warnings/counts retained. This file and audited_summary count them explicitly. Raw summaries max_gap=null may indicate nonfinite gaps, never zero error.'))
    with (H/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    assert not bound_violations
    # Paired grid claims, not prevalence estimates.
    for q in summaries:
        if q['gtol']==1e-6 and q['angle_change']==0:print(json.dumps(q))
    print('CONE',json.dumps({k:z for k,z in a.items() if k!='records'}))
    print('RECOVERY',len(all_recovery_errors),max(all_recovery_errors))
    print('FAILURE_RECORDS',len(failure_records))

if __name__=='__main__':run()
