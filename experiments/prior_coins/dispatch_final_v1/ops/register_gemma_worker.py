"""Resolve an allocated worker by authenticated pod identity, then preflight."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import keys
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write

ROOT=Path(__file__).resolve().parents[4]
ART=ROOT/'artifacts/aft_grid_8192_balanced_v2'
SKILL=Path('/root/.codex/skills/runpod-spinup')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',required=True);a=ap.parse_args()
    r=json.loads((ART/'deploy'/f'{a.worker}.json').read_text())
    env=dict(os.environ,RUNPOD_API_KEY=keys()[r['account']],
             PATH=str(ROOT/'artifacts/aft_size_mixture_v1/ops/bin')+':'+os.environ['PATH'])
    env.pop('SSH_AUTH_SOCK',None)
    matched=None
    for attempt in range(40):
        result=subprocess.run(['runpodctl','pod','get',r['pod_id']],env=env,capture_output=True,text=True,timeout=30)
        if result.returncode==0:
            info=json.loads(result.stdout);s=info.get('ssh',{})
            if s.get('ip') and s.get('port'):
                # TCP open is not identity: authenticate and verify the actual pod.
                for port in (s['port'],s['port']-1,s['port']+1):
                    ssh=['env','-u','SSH_AUTH_SOCK','ssh','-o','IdentitiesOnly=yes','-o','BatchMode=yes',
                         '-o','ConnectTimeout=4','-o','StrictHostKeyChecking=accept-new','-i','/root/.ssh/id_ed25519',
                         '-p',str(port),'root@'+s['ip'],'source /etc/rp_environment; printenv RUNPOD_POD_ID']
                    check=subprocess.run(ssh,capture_output=True,text=True,timeout=10)
                    if check.returncode==0 and check.stdout.strip()==r['pod_id']:
                        matched=(s['ip'],port,info);break
        if matched:break
        time.sleep(5)
    if not matched:raise RuntimeError('No authenticated endpoint; preserve pod and inspect')
    ip,port,info=matched
    entry=dict(pod_id=r['pod_id'],name=r['name'],account=r['account'],ip=ip,port=port,
               hourly=info['costPerHr'],gpu='H100' if r['model']=='12b' else 'H200',
               disk_gb=300 if r['model']=='12b' else 500)
    # Keep owned allocation visible even if preflight fails.
    lock=(ART/'catalog.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    catalog=ART/'PRODUCTION_PODS.json';pods=json.loads(catalog.read_text())
    if a.worker in pods and pods[a.worker]['pod_id']!=r['pod_id']:raise RuntimeError('Worker identity conflict')
    pods[a.worker]=entry;write(catalog,pods)
    fcntl.flock(lock,fcntl.LOCK_UN)
    subprocess.run(['python3',str(SKILL/'_ssh_alias.py'),'add',r['name'],r['pod_id'],ip,str(port)],check=True,env=env)
    with (ART/'deploy'/f'{a.worker}.preflight.log').open('w') as stream:
        result=subprocess.run([str(SKILL/'pod-preflight.sh'),r['pod_id'],'12.6'],
                              stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=180)
    if result.returncode:raise RuntimeError('Preflight failed; do not provision; inspect preflight log')
    write(ART/'deploy'/f'{a.worker}.ready.json',entry)
    print(a.worker,'PREFLIGHT PASSED',json.dumps(entry),flush=True)


if __name__=='__main__':main()
