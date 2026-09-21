"""Fresh SSH + independent immutable-HF audit of one completed Gemma cell."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from huggingface_hub import HfApi
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import verify
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import SAVES, write
from experiments.prior_coins.dispatch_final_v1.ops.provision_relocated_repair import connection


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True)
    p.add_argument('--job-index',type=int,required=True)
    p.add_argument('--root',help='Explicit remote queue root; do not follow ACTIVE_ROOT during a continuation handoff')
    a=p.parse_args()
    base=Path('artifacts/gemma_aft_halfpct_18workers_v1' if '-half' in a.worker else 'artifacts/aft_grid_8192_balanced_v2')
    pod=json.loads((base/'PRODUCTION_PODS.json').read_text())[a.worker]
    root=a.root or pod.get('root','/workspace/gemma-grid/'+a.worker)
    remote=f'''
import json
from pathlib import Path
root=Path({root!r})
if {a.root is None!r} and (root/'ACTIVE_ROOT.json').exists():root=Path(json.loads((root/'ACTIVE_ROOT.json').read_text())['root'])
worker=json.loads((root/'WORKER.json').read_text())
job=worker['plan']['workers'][worker['worker']]['jobs'][{a.job_index}]
dest=root/'cells'/job['id']
assert (dest/'COMPLETE.json').exists()
assert (dest/'TRAIN_COMPLETE.json').exists() and (dest/'EVAL_COMPLETE.json').exists()
receipts={{p.stem:json.loads(p.read_text()) for p in (dest/'receipts').glob('*.json')}}
print(json.dumps(dict(root=str(root),job=job,version=worker['plan']['version'],receipts=receipts)))
'''
    result=subprocess.run(connection(pod)+['python3 -'],input=remote,text=True,capture_output=True,timeout=40)
    assert result.returncode==0,result.stderr
    result=json.loads(result.stdout);receipts=result['receipts']
    required={'complete','provenance','eval-step256','eval-step512',*[f'checkpoint-{s}' for s in SAVES]}
    assert required.issubset(receipts),required-set(receipts)
    # The initial pilot published inputs in its final provenance commit, before
    # separate incremental inputs receipts were added. Check actual file coverage.
    covered={name for receipt in receipts.values() for name in receipt['files']}
    assert {'inputs.json','parent.json','IDENTITY.json','train/axolotl.yaml'}.issubset(covered)
    assert all(len(receipts[f'eval-step{s}']['files'])==21 for s in (256,512))
    api=HfApi();prefix=f"followups/{result['version']}/{result['job']['id']}"
    for label,r in receipts.items():
        assert r['prefix']==prefix
        verify(api,r['repo'],r['prefix'],r['commit'],r['files'])
    result.update(worker=a.worker,pod_id=pod['pod_id'],verified_at=time.time())
    out=base/'verified-cells'/a.worker/(result['job']['id'].replace('/','__')+'.json')
    write(out,result)
    print(a.worker,result['job']['id'],'VERIFIED',len(receipts),'receipts',receipts['complete']['commit'],flush=True)


if __name__=='__main__':main()
