"""Regenerate selected inputs and fit a verified original-image subset."""
from pathlib import Path
import argparse,importlib.util,json,shutil
SOURCE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--acquired',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
raw=a.acquired.resolve();out=a.output.resolve()
if out.exists() and any(out.iterdir()):raise SystemExit('Choose a new or empty output directory.')
(out/'sources').mkdir(parents=True,exist_ok=True)
for dataset in ['DeepSfP','PolarNS','PolarBurstSR']:
    shutil.copyfile(raw/'sources'/f'{dataset}_ACQUISITION.json',out/'sources'/f'{dataset}_ACQUISITION.json')
spec=importlib.util.spec_from_file_location('real_pilot',SOURCE/'run_real.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_inputs=m.inputs
m.P=raw
streams,metadata=original_inputs()
m.P=out;m.inputs=lambda:(streams,metadata);m.main()
reference=json.loads((SOURCE/'SUMMARY.json').read_text());actual=json.loads((out/'SUMMARY.json').read_text())
assert actual['inputs_sha256']==reference['inputs_sha256'],'Acquired source inputs do not match the pinned reference.'
print('Pinned original observations regenerated and fitted.')
