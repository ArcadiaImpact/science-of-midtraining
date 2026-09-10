"""Read-only live training + immutable HF checkpoint proof for rollout gate."""
import argparse
import json
from pathlib import Path
import subprocess
from huggingface_hub import HfApi
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import verify
from experiments.prior_coins.dispatch_final_v1.ops.provision_relocated_repair import connection

ART=Path('artifacts/aft_grid_8192_balanced_v2')
REMOTE=r'''
import json,pathlib,subprocess,math,ast
root=pathlib.Path(ROOT)
progress=json.loads((root/'train-progress.json').read_text())
receipt=json.loads((root/'receipts/checkpoint-4.json').read_text())
losses=[]
for line in (root/'train.log').read_text().splitlines():
 if line.startswith("{'loss':"):
  row=ast.literal_eval(line); losses.append(float(row['loss']))
if progress['step']<4 or not losses or not all(math.isfinite(x) for x in losses):raise RuntimeError('No finite training evidence')
if subprocess.run(['pgrep','-f','axolotl.cli.train.*'+str(root)],capture_output=True).returncode:raise RuntimeError('Training child missing')
print(json.dumps(dict(progress=progress,receipt=receipt,last_loss=losses[-1])))
'''


def main():
    global ART
    ap=argparse.ArgumentParser();ap.add_argument('--worker',required=True);a=ap.parse_args()
    half='-half' in a.worker
    if half: ART=Path('artifacts/gemma_aft_halfpct_18workers_v1')
    pod=json.loads((ART/'PRODUCTION_PODS.json').read_text())[a.worker]
    repair=pod.get('phase')=='repair'
    plan=json.loads((ART/('plan.json' if half else 'repair-plan-52cells.json' if repair else 'grid-plan-12workers.json')).read_text())
    logical=pod.get('logical_worker',a.worker)
    job=plan['workers'][logical]['jobs'][0]['id']
    root=pod.get('root','/workspace/gemma-grid/'+a.worker)+'/cells/'+job
    cmd=connection(pod)+['python3 -']
    r=subprocess.run(cmd,input='ROOT='+repr(root)+'\n'+REMOTE,text=True,capture_output=True,timeout=40)
    if r.returncode:raise RuntimeError('Gate not yet passed: '+r.stderr[-1000:])
    data=json.loads(r.stdout);receipt=data['receipt']
    verify(HfApi(),receipt['repo'],receipt['prefix'],receipt['commit'],receipt['files'])
    data.update(worker=a.worker,pod_id=pod['pod_id'],job=job)
    write(ART/'gates'/f'{a.worker}.json',data)
    print(a.worker,'GATE PASSED',data['progress']['step'],data['last_loss'],receipt['commit'])


if __name__=='__main__':main()
