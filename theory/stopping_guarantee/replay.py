"""Replay the immutable theory checks into a fresh directory."""
from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
assert not any(out.iterdir()),'Choose a new empty output directory.'
spec=importlib.util.spec_from_file_location('theory_checks',P/'checks.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
(out/'PROTOCOL.json').write_bytes((P/'results/PROTOCOL.json').read_bytes())
m.OUT=out;sys.argv=[str(P/'checks.py')];m.main()
expected=json.loads((P/'results/SUMMARY.json').read_text());actual=json.loads((out/'SUMMARY.json').read_text())
def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):assert np.isclose(a,b,rtol=1e-9,atol=1e-12),(a,b)
    else:assert a==b,(a,b)
compare(expected,actual)
ref=np.load(P/'results/AD_REFERENCE.npz');now=np.load(out/'AD_REFERENCE.npz')
assert ref.files==now.files
assert all(np.array_equal(ref[k],now[k]) for k in ref.files)
check=dict(scientific_summary_reproduced=True,AD_input_arrays_identical=True,
           objectives=actual['cases']['objectives'],derivative_error_checks=actual['cases']['input_error_checks'],
           frozen_endpoints=actual['frozen_endpoint_audit']['endpoints'])
(out/'REPRODUCTION_CHECK.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check))
