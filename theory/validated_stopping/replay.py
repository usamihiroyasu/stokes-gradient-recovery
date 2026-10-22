"""Reproduce deterministic interval records, keeping the frozen run intact."""
from pathlib import Path
import argparse,json,subprocess,sys

P=Path(__file__).resolve().parent


def without_runtime(x):
    if isinstance(x,dict):
        return {k:without_runtime(v) for k,v in x.items() if k not in ['elapsed_seconds','reference_seconds','environment']}
    if isinstance(x,list):return [without_runtime(v) for v in x]
    return x


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    subprocess.run([sys.executable,str(P/'run.py'),'--output',str(args.output)],check=True)
    identical={}
    for name in ['ENDPOINTS.json','REFERENCE_INTERVALS.json','STRESS.json']:
        identical[name]=(P/'results'/name).read_bytes()==(args.output/name).read_bytes()
        assert identical[name],name
    expected=json.loads((P/'results/SUMMARY.json').read_text());actual=json.loads((args.output/'SUMMARY.json').read_text())
    assert without_runtime(expected)==without_runtime(actual)
    receipt=dict(deterministic_records_byte_identical=identical,scientific_summary_identical=True,timings_and_environment_excluded=True)
    (args.output/'REPRODUCTION_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
