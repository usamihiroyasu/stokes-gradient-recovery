"""Read selected members of public ZIPs using validated HTTP byte ranges."""
import collections,hashlib,io,json,struct,urllib.request,zipfile,zlib
from pathlib import Path

class RangeFile:
    def __init__(self,url,size,block=2**20):
        self.url=url;self.size=size;self.position=0;self.block=block;self.cache=collections.OrderedDict();self.ledger=[]
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def tell(self):return self.position
    def seekable(self):return True
    def _get(self,start,end):
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','User-Agent':'Stokes-research-reproduction/1.0'})
        with urllib.request.urlopen(req,timeout=45) as r:
            cr=r.headers.get('Content-Range');expected=f'bytes {start}-{end}/{self.size}'
            if r.status!=206 or cr!=expected:raise RuntimeError(f'Range not honored: {r.status} {cr}; expected {expected}')
            raw=r.read(end-start+2)
            assert len(raw)==end-start+1
            self.ledger.append({'start':start,'end':end,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
            return raw
    def read(self,n=-1):
        if n<0:n=self.size-self.position
        n=min(n,self.size-self.position)
        if n<=0:return b''
        assert n<128*2**20,'Refuse an unexpectedly large single read.'
        result=[]
        while n:
            index=self.position//self.block;start=index*self.block
            if index not in self.cache:
                self.cache[index]=self._get(start,min(self.size-1,start+self.block-1))
                if len(self.cache)>8:self.cache.popitem(last=False)
            data=self.cache[index];offset=self.position-start;take=min(n,len(data)-offset)
            assert take>0
            result.append(data[offset:offset+take]);self.position+=take;n-=take
        return b''.join(result)

DATASETS={
'DeepSfP':('1EtjfMTfpanJotH92GFz300X_ZEmGXuqr',4956684332),
'PolarNS':('1e10inseqNhd1oL2TMThGYMiEiX7y4lNW',14202427825),
'PolarBurstSR':('1qWKMJdIS11RbvN9DMIo-QtCSsi1yCGv6',5630323009)}

def remote(dataset):
    ident,size=DATASETS[dataset]
    f=RangeFile('https://drive.usercontent.google.com/download?id='+ident+'&export=download&confirm=t',size)
    return f,zipfile.ZipFile(f)

if __name__=='__main__':
    P=Path(__file__).resolve().parent/'sources'
    for dataset in DATASETS:
        f,z=remote(dataset)
        rows=[{'name':v.filename,'uncompressed':v.file_size,'compressed':v.compress_size,'crc32':f'{v.CRC:08x}','offset':v.header_offset,'method':v.compress_type,'flags':v.flag_bits} for v in z.infolist()]
        (P/f'{dataset}_ZIP_INDEX.json').write_text(json.dumps({'dataset':dataset,'size':f.size,'files':rows,'range_ledger':f.ledger},indent=2)+'\n')
        print(json.dumps({'dataset':dataset,'entries':len(rows),'metadata_bytes_downloaded':sum(v['bytes'] for v in f.ledger),'first_files':rows[:12]}),flush=True)
