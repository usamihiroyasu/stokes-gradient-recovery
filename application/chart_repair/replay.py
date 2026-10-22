"""Replay the frozen study and retain every internal native solver status."""
from pathlib import Path
import argparse,importlib.util,json
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
assert not any(out.iterdir()),'Choose a new empty output directory.'
spec=importlib.util.spec_from_file_location('frozen_chart',P/'run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
(out/'PROTOCOL.json').write_bytes((P/'frozen/PROTOCOL.json').read_bytes())
native=m.c.solve;calls=[]
def logged(x,y,method,tol):
    z=native(x,y,method,tol)
    calls.append(dict(method=method,success=z['success'],status=z['status'],message=z['message'],
                      iterations=z['nit'],counts=z['counts'],warnings=z['warnings']))
    return z
m.c.solve=logged
m.run(out)
expected=json.loads((P/'frozen/SUMMARY.json').read_text());actual=json.loads((out/'SUMMARY.json').read_text())
def scientific(x):
    if isinstance(x,dict):return {k:scientific(v) for k,v in x.items() if k not in ['elapsed_seconds','protocol_sha256']}
    if isinstance(x,list):return [scientific(v) for v in x]
    return x
assert scientific(expected)==scientific(actual)
receipt=dict(scientific_summary_exact_match=True,timing_excluded=True,
             internal_native_invocations=len(calls),
             native_failures=sum(not x['success'] for x in calls),
             warning_invocations=sum(bool(x['warnings']) for x in calls),
             nonfinite_queries=sum(x['counts']['nonfinite_calls'] for x in calls))
(out/'ALL_NATIVE_CALLS.json').write_text(json.dumps(calls,indent=2)+'\n')
(out/'REPRODUCTION_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
