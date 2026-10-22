"""Replay the fixed pilot in a fresh directory while retaining its reference records."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
SOURCE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output.resolve()
if out.exists() and any(out.iterdir()):raise SystemExit('Choose a new or empty output directory.')
out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('fresnel_pilot',SOURCE/'run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.HERE=out;m.main()
reference=json.loads((SOURCE/'SUMMARY.json').read_text());actual=json.loads((out/'SUMMARY.json').read_text())
assert actual['inputs_sha256']==reference['inputs_sha256']
for expected,got in zip(reference['arms'],actual['arms']):
    for key in expected:
        if key=='optimizer_seconds':continue
        if isinstance(expected[key],float):assert np.isclose(expected[key],got[key],rtol=1e-8,atol=1e-12),(key,expected[key],got[key])
        else:assert expected[key]==got[key],(key,expected[key],got[key])
# Replay the corrected alternative-map control on the same saved inputs.
x_fold=np.array([0.,.2,-.1]);s0,_=m.c.geometry(x_fold);pol=s0[1:]/s0[0]
x0=np.r_[0.,pol/np.sqrt(1-pol@pol)];matched=[]
for inp in json.loads((SOURCE/'inputs.json').read_text()):
    x=x0.copy()
    for y in inp['observations']:
        y=np.array(y);_,minimum=m.c.cone_solution(y)
        final=m.assess(m.mono_solve(x,y),y,minimum,True);x=np.array(final['x']);matched.append(final)
expected=json.loads((SOURCE/'matched_monotone_control.json').read_text())['summary']
assert sum(r['false_success'] for r in matched)==expected['false_success']
assert sum(not r['success'] for r in matched)==expected['solver_nonacceptances']
assert np.isclose(max(r['gap'] for r in matched),expected['max_gap'],rtol=1e-8,atol=1e-12)
(out/'REPRODUCTION_CHECK.json').write_text(json.dumps({'same_inputs':True,'scientific_summaries_match':True,'matched_control_reproduced':True,'timing_excluded_from_comparison':True},indent=2)+'\n')
print('Reference inputs and scientific summaries reproduced.')
