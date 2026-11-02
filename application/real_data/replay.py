"""Replay frozen selected observations without downloading full image payloads."""
from pathlib import Path
import argparse,importlib.util,json,shutil,sys
import numpy as np
SOURCE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
out=args.output.resolve()
if out.exists() and any(out.iterdir()):raise SystemExit('Choose a new or empty output directory.')
(out/'sources').mkdir(parents=True,exist_ok=True)
for dataset in ['DeepSfP','PolarNS','PolarBurstSR']:
    shutil.copyfile(SOURCE/'sources'/f'{dataset}_ACQUISITION.json',out/'sources'/f'{dataset}_ACQUISITION.json')
spec=importlib.util.spec_from_file_location('real_pilot',SOURCE/'run_real.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
streams=json.loads((SOURCE/'INPUTS.json').read_text());meta=json.loads((SOURCE/'DATA_SCHEMA.json').read_text())
for stream in streams:
    stream['observations']=np.array(stream['observations'],dtype=np.float64);stream['coordinates']=np.array(stream['coordinates'],dtype=int)
m.P=out;m.inputs=lambda:(streams,meta);m.main()
reference=json.loads((SOURCE/'SUMMARY.json').read_text());actual=json.loads((out/'SUMMARY.json').read_text())
assert reference['inputs_sha256']==actual['inputs_sha256']
assert reference['counts']==actual['counts'] and reference['optimizer_fits']==actual['optimizer_fits']
for expected,got in zip(reference['summary'],actual['summary']):
    for key,val in expected.items():
        if isinstance(val,float):assert np.isclose(val,got[key],rtol=1e-8,atol=1e-12),(key,val,got[key])
        else:assert val==got[key],(key,val,got[key])
(out/'REPRODUCTION_CHECK.json').write_text(json.dumps({'identical_selected_observations':True,'scientific_counts_and_summary_values_match':True,'raw_image_payload_download_required':False,'fits':actual['optimizer_fits']},indent=2)+'\n')
print('Frozen real-observation audit reproduced.')
