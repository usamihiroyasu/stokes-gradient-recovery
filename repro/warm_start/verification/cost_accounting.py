"""Reconcile recorded optimizer queries and deterministic analysis overhead.

No optimizer execution and no changes to registered result files.
"""
from pathlib import Path
import json,csv
H=Path(__file__).resolve().parent;R=H.parent

def main():
    main=json.loads((H/'main_results.json').read_text())['records'];p=json.loads((H/'precision_results.json').read_text())
    source=json.loads((R/'strategy/warm_start_probe.json').read_text())['records'];unique={}
    for row in main:
        key=(row['method'],row['gtol'],tuple(source[row['case']]['previous_observations']))
        unique[key]=row['previous']
    def counts(rows):return {k:sum(x['counts'][k] for x in rows) for k in ['function','gradient','hessian','nonfinite_calls']}
    sci=dict(previous_unique_runs=len(unique),previous_optimizer_queries=counts(list(unique.values())),baseline_optimizer_queries=counts([r['baseline'] for r in main]),unconditional_restart_optimizer_queries=counts([r['always_restart'] for r in main]),analysis_head_function_gradient_pairs=dict(radial_trials=3*len(main),endpoint_diagnostic_state=len(main),reference_gap_rechecks=2*len(main)),analysis_hessian_queries=len(main),analytic_cone_solves=len(main),gate_hessians_required_after_radius_guard=sum(r['diagnostic']['near_fold'] for r in main))
    rs=p['records'];unique={}
    for row in rs:unique[(row['dtype'],tuple(source[row['case']]['previous_observations']))]=row['previous']
    def tc(rows):return dict(function=sum(r['function'] for r in rows),gradient=sum(r['gradient'] for r in rows),native_terminal_function_gradient_checks=len(rows),nonfinite_calls=sum(r['nonfinite_calls'] for r in rows))
    torch=dict(previous_unique_runs=len(unique),previous_optimizer_queries=tc(list(unique.values())),baseline_optimizer_queries=tc([r['baseline'] for r in rs]),unconditional_restart_optimizer_queries=tc([r['always_restart'] for r in rs]),float64_analysis_function_gradient_pairs=dict(preliminary_gap_recheck=len(rs),radial_trials=3*len(rs),final_gap_rechecks=2*len(rs)),float64_analysis_hessian_queries=len(rs),analytic_cone_solves=len(rs))
    ledger=dict(schema='stokes-query-accounting-v1',semantics='Function and gradient columns count jointly acquired calls, not two interchangeable work units. Hessian queries are full analytic or AD 3x3 matrices. Final checks and shadow coverage analysis are separated from optimizer calls.',scipy=sci,pytorch=torch,separate_procedures=dict(preflight_frozen_endpoint_AD=60,physical_SLSQP_audits=180,scipy_mixed_dtype_probe=1),deployment_note='Gated restart and unconditional restart reuse the same deterministic restart result when triggered. Required gate Hessian counts apply a radius guard before querying; actual coverage logging computes all endpoint Hessians. No end-to-end runtime comparison is inferred.')
    (H/'COST_ACCOUNTING.json').write_text(json.dumps(ledger,indent=2)+'\n')
    rows=[]
    for q in p['summary']:
        row={k:q[k] for k in ['dtype','angle_change','cases','previous_failures','triggers','detected_false_success','accurate_success_triggered']}
        for arm in ['baseline','always_restart','triggered_restart']:
            row.update({arm+'_'+k:v for k,v in q[arm].items()})
        rows.append(row)
    with (H/'precision_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(ledger,indent=2))

if __name__=='__main__':main()
