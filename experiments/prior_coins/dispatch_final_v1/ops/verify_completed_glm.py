"""Read-only SSH and immutable-HF audit for completed A1 GLM row-dose cells."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import HfApi
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import verify
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write

REMOTE = r'''
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/workspace/scimt/experiments/prior_coins/dispatch_final_v1/aft_size_mixture_v1')
from run import verify_adapters
root=Path(ROOT)/CELL
assert json.loads((root/'TRAIN_COMPLETE.json').read_text())['steps']==5120
assert (root/'EVAL_COMPLETE.json').exists()
verify_adapters(root)
scores=json.loads((root/'scored.json').read_text())
assert scores['cell']==CELL and set(scores['epochs'])=={'2560','5120'}
for step,sets in scores['epochs'].items():
 assert len(sets)==18
 for name,score in sets.items():
  records=[json.loads(line) for line in (root/'eval'/f'{CELL}-step{step}'/(name+'.jsonl')).read_text().splitlines()]
  assert len(records)==score['n'] and len({r['id'] for r in records})==len(records)>0
def record(path):
 size=path.stat().st_size;sha=hashlib.sha256();git=hashlib.sha1(b'blob '+str(size).encode()+bytes([0]))
 with path.open('rb') as handle:
  for block in iter(lambda:handle.read(8*1024*1024),b''):sha.update(block);git.update(block)
 return dict(size=size,sha256=sha.hexdigest(),git_blob=git.hexdigest())
files={str(p.relative_to(root)):record(p) for p in root.rglob('*') if p.is_file() and
 (p.relative_to(root).parts[0] in ('adapters','eval','health') or
 (p.parent==root and p.suffix in ('.json','.yaml','.log') and p.name!='PUBLISHED.json'))}
print(json.dumps(dict(publication=json.loads((root/'PUBLISHED.json').read_text()),files=files,
 scores=scores,identity=json.loads((root/'IDENTITY.json').read_text()))))
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=('charter','coin','control'),required=True)
    p.add_argument('--cell',choices=('agreement','charter_1pct','coin_1pct'),required=True)
    a=p.parse_args()
    catalog=Path('experiments/prior_coins/dispatch_final_v1/ops/handrun_units.tsv')
    rows=[s.split('\t') for s in catalog.read_text().splitlines()
          if s.startswith(f'glm-aft81920/{a.arm}\t')]
    assert len(rows)==1
    row=rows[0]
    command=['env','-u','SSH_AUTH_SOCK','ssh','-o','IdentitiesOnly=yes','-o','BatchMode=yes',
             '-o','ConnectTimeout=10','-i','/root/.ssh/id_ed25519',row[3],'python3 -']
    result=subprocess.run(command,input=f'ROOT={row[5]!r}\nCELL={a.cell!r}\n'+REMOTE,
                          capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stderr
    evidence=json.loads(result.stdout);r=evidence['publication']
    version='aft-size-mixture-v1' if a.cell=='agreement' else 'aft-size-mixture-rows-v2'
    prefix=f'followups/{version}/{a.arm}/{a.cell}'
    verify(HfApi(),r['repo'],prefix,r['commit'],evidence['files'])
    evidence.update(prefix=prefix,pod_id=row[2],verified_at=time.time())
    write(Path('artifacts/aft_size_mixture_v1/verified-cells')/f'{a.arm}-{a.cell}.json',evidence)
    print(a.arm,a.cell,'VERIFIED',len(evidence['files']),'files',r['commit'],flush=True)


if __name__=='__main__':main()
