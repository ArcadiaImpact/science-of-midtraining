"""Stage exact missing GLM eval wheels via the coordinator's faster PyPI route.

No process signals or environment installs. SHA256 checked against PyPI metadata.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,re,subprocess,urllib.request
from pathlib import Path
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write
from experiments.prior_coins.dispatch_final_v1.ops.provision_relocated_repair import connection

BASE=Path('artifacts/glm_aft_8192_queued_v2/deployment/control-wheel-recovery')
PINS={'torch':'2.10.0','vllm':'0.19.1','flashinfer-cubin':'0.6.6',
 'nvidia-cublas-cu12':'12.8.4.1','nvidia-cudnn-cu12':'9.10.2.21',
 'nvidia-cusolver-cu12':'11.7.3.90','nvidia-cusparselt-cu12':'0.7.1',
 'nvidia-nccl-cu12':'2.27.5','nvidia-cusparse-cu12':'12.5.8.93'}

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');args=parser.parse_args()
    BASE.mkdir(exist_ok=True,parents=True)
    pod=json.loads(Path('artifacts/glm_aft_8192_queued_v2/deployment/A3-glm-1c-control.json').read_text())
    assert pod['pod_id']=='ay0lzjaqrjaoic'
    pins=PINS
    uploaded=set()
    if (BASE/'MANIFEST.json').exists():uploaded={x['filename'] for x in json.loads((BASE/'MANIFEST.json').read_text())['files']}
    if args.all:
        subprocess.run(connection(pod,True)+['root@'+pod['ip']+':/workspace/glm-control-recovery-wheels/resolved-with-hashes.txt',str(BASE/'resolved-with-hashes.txt')],check=True,timeout=40)
        pins=dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)',(BASE/'resolved-with-hashes.txt').read_text(),re.M))
        assert len(pins)==182 and all(pins[k]==v for k,v in PINS.items())
    tags=list(sys_tags());rank={t:i for i,t in enumerate(tags)}
    def fetch(item):
        name,version=item
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30) as r:data=json.load(r)
        candidates=[]
        for f in data['urls']:
            if not f['filename'].endswith('.whl'):continue
            _,_,_,ts=parse_wheel_filename(f['filename'])
            scores=[rank[t] for t in ts if t in rank]
            if scores:candidates.append((min(scores),f))
        assert candidates,name
        f=min(candidates,key=lambda x:x[0])[1];dest=BASE/f['filename']
        if not dest.exists():
            temp=dest.with_suffix('.part')
            subprocess.run(['curl','--fail','--location','--retry','2','--max-time','240',
                '--silent','--show-error','--output',str(temp),f['url']],check=True)
            assert temp.stat().st_size==f['size'] and digest(temp)==f['digests']['sha256']
            temp.rename(dest)
        assert dest.stat().st_size==f['size'] and digest(dest)==f['digests']['sha256']
        if dest.name not in uploaded:
            subprocess.run(connection(pod,True)+[str(dest),'root@'+pod['ip']+':/workspace/glm-control-recovery-wheels/'+dest.name],check=True,timeout=240)
        print('STAGED',name,version,f['size'],flush=True)
        return dict(name=name,version=version,filename=dest.name,size=f['size'],sha256=f['digests']['sha256'],url=f['url'])
    subprocess.run(connection(pod)+['mkdir -p /workspace/glm-control-recovery-wheels'],check=True,timeout=30)
    with ThreadPoolExecutor(8 if args.all else 4) as pool:rows=list(pool.map(fetch,pins.items()))
    write(BASE/'MANIFEST.json',dict(pod_id=pod['pod_id'],files=rows))
    subprocess.run(connection(pod,True)+[str(BASE/'MANIFEST.json'),'root@'+pod['ip']+':/workspace/glm-control-recovery-wheels/MANIFEST.json'],check=True,timeout=30)
    print('ALL WHEELS STAGED; remote verification and guarded retry still required',flush=True)

if __name__=='__main__':main()
