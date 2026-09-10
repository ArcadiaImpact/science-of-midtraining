"""User-approved GLM launch, with explicit ownership and conservative fleet cap.

The skill's create scripts cannot request minimum host RAM. Only that creation
gap uses the existing account-explicit GraphQL helper; preflight/status/aliases
use the skill. No stop/delete API. Immutable prepared science stays untouched.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import get_token,HfApi
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import keys,inventory,query
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind,write,sha
from experiments.prior_coins.dispatch_final_v1.ops.provision_relocated_repair import connection
from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.run import ready_inputs

BASE=Path('artifacts/glm_aft_8192_queued_v2')
OPS=BASE/'deployment'
SKILL=Path('/root/.codex/skills/runpod-spinup')

def env_for(account):
    env=dict(os.environ,RUNPOD_API_KEY=keys()[account],PATH='/tmp/glm-pause-cli.g2Zd6U:'+os.environ['PATH'])
    env.pop('SSH_AUTH_SOCK',None)
    return env

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True)
    p.add_argument('--phase',choices=('create','provision'),required=True);a=p.parse_args()
    plan=json.loads((BASE/'plan.json').read_text());ready_inputs(BASE,plan)
    spec=plan['workers'][a.worker];account=spec['account'];creds=keys()
    receipt=OPS/f'{a.worker}.json';pending=OPS/f'{a.worker}.pending.json'
    name='glm-8192-'+a.worker.lower()+'-20260908'
    if a.phase=='create':
        fleet=inventory(creds)
        provider=json.loads(subprocess.check_output(['runpodctl','me'],env=env_for(account),text=True))
        total=max(float(provider['currentSpendPerHr']),sum(float(p['costPerHr']) for p in fleet[account]['pods']))
        if total+18.70>min(80,provider['spendLimit']):
            raise RuntimeError(f'Provider per-account cap: {account} {total}+18.70 > 80')
        if receipt.exists() or pending.exists() or any(p['name']==name for v in fleet.values() for p in v['pods']):
            raise RuntimeError('Existing/ambiguous allocation; inspect, never duplicate')
        bind(pending,dict(worker=a.worker,account=account,name=name,prior_fleet_hourly=total,
             cap_per_account=80,provider_limit=provider['spendLimit'],user_authorized=True,plan_sha256=sha(BASE/'plan.json')))
        public=Path('/root/.ssh/id_ed25519.pub').read_text().strip()+'\n'+subprocess.check_output(
            ['ssh-keygen','-y','-f','/root/.runpod/ssh/runpodctl-ssh-key'],text=True).strip()
        body='''mutation { podFindAndDeployOnDemand(input: {
          cloudType: SECURE, gpuCount: 4, gpuTypeId: "NVIDIA H200",
          templateId: "runpod-torch-v280", containerDiskInGb: 2000,
          volumeInGb: 0, minMemoryInGb: 1000,
          allowedCudaVersions: ["12.8","12.9","13.0","13.1"],
          ports: "22/tcp,8888/http", startSsh: true, supportPublicIp: true,
          env: [{key: "PUBLIC_KEY", value: SSH_PUBLIC_VALUE}], name: POD_NAME_VALUE
        }) { id machineId costPerHr } }'''.replace('SSH_PUBLIC_VALUE',json.dumps(public)).replace('POD_NAME_VALUE',json.dumps(name))
        result=query(creds[account],body).get('podFindAndDeployOnDemand')
        if not result or not result.get('id'):raise RuntimeError('Unconfirmed allocation: reconcile pending intent')
        bind(receipt,dict(worker=a.worker,account=account,name=name,pod_id=result['id'],
            hourly=result['costPerHr'],gpu='H200',gpu_count=4,disk_gb=2000,min_ram_gb=1000,
            cloud='SECURE',created_at=time.time(),machine_id=result.get('machineId')))
        print(receipt.read_text(),flush=True);return
    pod=json.loads(receipt.read_text());env=env_for(account)
    assert not (OPS/f'{a.worker}.launched.json').exists(),'Already launched; inspect'
    live=[p for p in inventory(creds)[account]['pods'] if p['id']==pod['pod_id']]
    assert len(live)==1 and live[0]['name']==pod['name']
    for attempt in range(40):
        r=subprocess.run(['python3',str(SKILL/'_resolve_ssh.py'),pod['pod_id'],'--quiet'],env=env,capture_output=True,text=True,timeout=30)
        if len(r.stdout.split())==2:
            ip,port=r.stdout.split();pod.update(ip=ip,port=int(port))
            r=subprocess.run(connection(pod)+['source /etc/rp_environment; printenv RUNPOD_POD_ID'],capture_output=True,text=True,timeout=15)
            if r.returncode==0 and r.stdout.strip()==pod['pod_id']:break
        time.sleep(5)
    else:raise RuntimeError('No authenticated SSH; preserve allocation and investigate')
    write(receipt,pod)
    shared_alias_lock=Path('artifacts/gemma_aft_halfpct_18workers_v1/deployment/allocation.lock')
    with (shared_alias_lock if shared_alias_lock.exists() else OPS/'alias.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(['python3',str(SKILL/'_ssh_alias.py'),'add',name,pod['pod_id'],pod['ip'],str(pod['port'])],env=env,check=True)
    with (OPS/f'{a.worker}.preflight.log').open('w') as log:
        r=subprocess.run([str(SKILL/'pod-preflight.sh'),pod['pod_id'],'12.8'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    assert r.returncode==0,'Preflight failed; inspect log'
    subprocess.run(connection(pod)+['test ! -e /workspace/GLM_LAUNCHED.json; mkdir -p /workspace/scimt /workspace/glm-1c-prepared; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'],check=True,timeout=90)
    files=[(BASE/'base-code.tar.gz','/workspace/base-code.tar.gz'),(BASE/'code-overlay.tar.gz','/workspace/code-overlay.tar.gz'),(BASE/'prepared-inputs.tar.gz','/workspace/prepared-inputs.tar.gz')]
    for local,remote in files:
        subprocess.run(connection(pod,True)+[str(local),'root@'+pod['ip']+':'+remote],check=True,timeout=300)
    subprocess.run(connection(pod,True)+[str(Path(__file__).with_name('glm_deployment_git.py')),
        'root@'+pod['ip']+':/workspace/glm_deployment_git.py'],check=True,timeout=60)
    # Refuse any historical/duplicate output namespace before starting work.
    api=HfApi()
    from huggingface_hub.errors import EntryNotFoundError
    for job in spec['jobs']:
        try:existing=list(api.list_repo_tree(plan['parent_repo'],path_in_repo=f"followups/{plan['version']}/{job['id']}"))
        except EntryNotFoundError:existing=[]
        assert not existing,'Existing HF output; reconcile, never overwrite'
    token=get_token();assert token
    cmd=('set -e; set -a; source /etc/rp_environment; set +a; read -r HF_TOKEN; export HF_TOKEN; '
         'export PATH=/root/.local/bin:$PATH; tar -xzf /workspace/base-code.tar.gz -C /workspace/scimt; '
         'tar -xzf /workspace/code-overlay.tar.gz -C /workspace/scimt; '
         'python3 /workspace/glm_deployment_git.py; '
         'tar -xzf /workspace/prepared-inputs.tar.gz -C /workspace/glm-1c-prepared; '
         'test ! -e /workspace/GLM_LAUNCHED.json; cp /workspace/glm-1c-prepared/READY.json /workspace/GLM_LAUNCHED.json; '
         'tmux new-session -d -s glm-aft "bash /workspace/scimt/experiments/prior_coins/dispatch_final_v1/glm_aft_repair_v1/setup.sh '+a.worker+' '+account+' >/workspace/glm-repair-worker.log 2>&1"; echo LAUNCHED')
    r=subprocess.run(connection(pod)+[cmd],input=token+'\n',capture_output=True,text=True,timeout=120)
    assert r.returncode==0,r.stderr
    write(OPS/f'{a.worker}.launched.json',dict(pod=pod,launched_at=time.time(),output=r.stdout,
          root='/workspace/glm-aft-2pct-repair-v1/'+a.worker,log='/workspace/glm-repair-worker.log'))
    subprocess.run([str(SKILL/'pod-status.sh'),pod['pod_id']],env=env,check=False,timeout=120)
    print('LAUNCHED',a.worker,pod['pod_id'],flush=True)

if __name__=='__main__':main()
