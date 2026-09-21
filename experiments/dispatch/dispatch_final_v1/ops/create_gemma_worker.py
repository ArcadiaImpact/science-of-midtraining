"""Create exactly one approved worker via the RunPod skill, with durable intent.

No provisioning or deletion. On ambiguous failure reconcile by name; never
repeat a create whose pending receipt has not been resolved.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from experiments.dispatch.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import keys, inventory
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import bind, write

ROOT=Path(__file__).resolve().parents[4]
ART=ROOT/'artifacts/aft_grid_8192_balanced_v2'
SKILL=Path('/root/.codex/skills/runpod-spinup')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True)
    p.add_argument('--extra-repair',action='store_true')
    p.add_argument('--cli-dir',type=Path);a=p.parse_args()
    plan=json.loads((ART/'grid-plan-12workers.json').read_text())
    if a.extra_repair:
        destinations=[f'A{account}-27b-r{slot}' for account in (2,3) for slot in (1,2,3)]
        sources=sorted(k for k,w in plan['workers'].items() if w['model']=='27b')
        assert a.worker in destinations
        logical_worker=sources[destinations.index(a.worker)]
        worker=dict(account=a.worker.split('-')[0],model='27b')
    else:
        worker=plan['workers'][a.worker]
    account=worker['account'];model=worker['model']
    credentials=keys();accounts=inventory(credentials)
    if a.extra_repair and sum(x['costPerHr'] for k in ('A2','A3') for x in accounts[k]['pods'])+4.59>60.001:
        raise RuntimeError('Combined A2/A3 $60/hour ceiling would be exceeded')
    name='gemma-grid-'+a.worker.lower()+'-20260907'
    receipt=ART/'deploy'/f'{a.worker}.json'
    pending=ART/'deploy'/f'{a.worker}.pending.json'
    if receipt.exists() or pending.exists() or any(x['name']==name for v in accounts.values() for x in v['pods']):
        raise RuntimeError('Existing/ambiguous deployment; reconcile before another create')
    bind(pending,dict(worker=a.worker,account=account,name=name))
    env=dict(os.environ,RUNPOD_API_KEY=credentials[account],
             PATH=str(a.cli_dir or ROOT/'artifacts/aft_size_mixture_v1/ops/bin')+':'+os.environ['PATH'])
    cmd=[str(SKILL/'create-pod-cuda.sh'),name,
         'NVIDIA H100 80GB HBM3' if model=='12b' else 'NVIDIA H200',
         '12.8,12.9,13.0,13.1','SECURE','runpod-torch-v280','1','300' if model=='12b' else '500']
    log=ART/'deploy'/f'{a.worker}.create.log'
    with log.open('w') as stream:
        result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=700)
    text=log.read_text();match=re.search(r'Pod ID:\s+(\w+)',text)
    if not match:
        raise RuntimeError(f'No confirmed ID; reconcile inventory and {log}; rc={result.returncode}')
    pod=match[1]
    # The standard skill does not inject our non-agent monitoring key. Install
    # it before transferring any data; this restarts the empty container.
    update=subprocess.run(['runpodctl','pod','update',pod,'--env',
        json.dumps({'PUBLIC_KEY':Path('/root/.ssh/id_ed25519.pub').read_text().strip()})],
        env=env,capture_output=True,text=True,timeout=60)
    if update.returncode:raise RuntimeError(f'{pod}: public-key update failed')
    write(receipt,dict(worker=a.worker,account=account,name=name,pod_id=pod,
                      model=model,create_returncode=result.returncode,needs_endpoint_resolution=True,
                      **(dict(logical_worker=logical_worker,phase='repair') if a.extra_repair else {})))
    print(json.dumps(json.loads(receipt.read_text())),flush=True)


if __name__=='__main__':main()
