"""Fetch and verify the pinned original image/statistic subset to a fresh folder."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,shutil
import acquire
SOURCE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
if out.exists() and any(out.iterdir()):raise SystemExit('Choose a new or empty output directory.')
(out/'sources').mkdir(parents=True,exist_ok=True)
for dataset in ['DeepSfP','PolarNS','PolarBurstSR']:
    shutil.copyfile(SOURCE/'sources'/f'{dataset}_ZIP_INDEX.json',out/'sources'/f'{dataset}_ZIP_INDEX.json')
acquire.P=out
with ThreadPoolExecutor(max_workers=3) as pool:receipts=list(pool.map(acquire.download,['DeepSfP','PolarNS','PolarBurstSR']))
for dataset in ['DeepSfP','PolarNS','PolarBurstSR']:
    expected=json.loads((SOURCE/'sources'/f'{dataset}_ACQUISITION.json').read_text())
    for member in expected['fetched_members']:
        path=out/'data'/dataset/member['member'];assert hashlib.sha256(path.read_bytes()).hexdigest()==member['sha256'],str(path)
(out/'PINNED_DATA_VERIFICATION.json').write_text(json.dumps({'original_members_verified':51,'matches_reference_sha256':True,'whole_archives_downloaded':False,'receipts':receipts},indent=2)+'\n')
