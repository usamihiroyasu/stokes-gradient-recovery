"""Fetch a predetermined, bounded subset of the authors' public archives."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,struct,zlib
from remote_zip import remote
P=Path(__file__).resolve().parent

def names_for(dataset):
    rows=json.loads((P/'sources'/f'{dataset}_ZIP_INDEX.json').read_text())['files']
    names=[r['name'] for r in rows]
    if dataset=='DeepSfP':
        return ['deepSfP_data/test_list.csv','deepSfP_data/README.txt']+['deepSfP_data/indoor/'+n+'.mat' for n in ['box_f','father_christmas_f','horse_l']]
    if dataset=='PolarNS':
        out=[]
        for prefix in ['object/','scene/']:
            chosen=sorted(n for n in names if n.startswith(prefix) and n.endswith('/img_stat.mat'))[0]
            out += [chosen,chosen.rsplit('/',1)[0]+'/capture_info.txt']
        return out
    return [f'test/{i:07d}/frame_{t:02d}.png' for i in range(3) for t in range(14)]


def download(dataset):
    f,z=remote(dataset);out=[]
    for name in names_for(dataset):
        v=z.getinfo(name);assert v.file_size<=100*2**20
        # The old DeepSfP ZIP's end-directory offset wraps at 2**32. Python's
        # prefix inference consequently adds 2**32 to its local offsets.
        # Resolve the location by matching the local signature and exact name.
        candidates=[v.header_offset] if dataset!='DeepSfP' else [v.header_offset%(2**32),v.header_offset%(2**32)+2**32]
        found=None
        for at in candidates:
            if not 0<=at<f.size-1024:continue
            head=f._get(at,at+1023)
            if head[:4]!=b'PK\x03\x04':continue
            nl,xl=struct.unpack_from('<HH',head,26)
            if head[30:30+nl]!=name.encode():continue
            found=at+30+nl+xl;break
        assert found is not None,('local header not found',name,candidates)
        assert found+v.compress_size<=f.size
        raw=f._get(found,found+v.compress_size-1) if v.compress_size else b''
        data=zlib.decompress(raw,-15) if v.compress_type==8 else raw
        assert len(data)==v.file_size and zlib.crc32(data)&0xffffffff==v.CRC
        dest=P/'data'/dataset/PurePosixPath(name)
        assert dest.resolve().is_relative_to((P/'data').resolve())
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(data).digest()
        else:dest.write_bytes(data)
        rec={'member':name,'bytes':len(data),'compressed_bytes':len(raw),'sha256':hashlib.sha256(data).hexdigest(),'compressed_sha256':hashlib.sha256(raw).hexdigest(),'crc32':f'{v.CRC:08x}','verified_local_header':at,'payload_range':[found,found+v.compress_size-1]}
        out.append(rec);print(json.dumps({'dataset':dataset,'member':name,'bytes':len(data)}),flush=True)
    receipt={'dataset':dataset,'archive_bytes':f.size,'source_url':f.url,'fetched_members':out,'range_ledger':f.ledger,'network_bytes':sum(r['bytes'] for r in f.ledger),'whole_archive_downloaded':False,'whole_archive_sha256_available':False}
    (P/'sources'/f'{dataset}_ACQUISITION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return {'dataset':dataset,'members':len(out),'member_bytes':sum(r['bytes'] for r in out),'network_bytes':receipt['network_bytes']}

if __name__=='__main__':
    plan=P/'ACQUISITION_PLAN.json'
    assert not plan.exists(),'Preserve the acquisition plan; use another run directory.'
    choices={d:names_for(d) for d in ['DeepSfP','PolarNS','PolarBurstSR']}
    plan.write_text(json.dumps({'registered_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':'DeepSfP: first three distinct indoor test objects in published test list; PolarNS: lexicographically first object and scene statistics; PolarBurstSR: first three released test crops, all 14 raw frames. No model outcomes inspected.','scope':'Local bounded research subset, no model training or external redistribution of raw images','files':choices},indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(download,choices))
    (P/'ACQUISITION_SUMMARY.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'completed':results}),flush=True)
