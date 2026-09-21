"""Finite, resumable provisioning of approved parent-pair workers on A2/A3.

Uses RunPod skill creation/preflight/status; never deletes or stops a pod.
Fresh per-account provider spend checks, serialized allocation and durable
intents prevent duplicate or over-budget allocation. No scheduler is installed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import EntryNotFoundError
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import keys, inventory
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind, write, sha
from experiments.prior_coins.dispatch_final_v1.gemma_halfpct_sharded import validate
from experiments.prior_coins.dispatch_final_v1.ops.provision_relocated_repair import connection

BASE = Path('artifacts/gemma_aft_halfpct_18workers_v1')
OPS = BASE / 'deployment'
SKILL = Path('/root/.codex/skills/runpod-spinup')
CLI = '/tmp/glm-pause-cli.g2Zd6U'


def environment(account):
    env = dict(os.environ, RUNPOD_API_KEY=keys()[account], PATH=CLI+':'+os.environ['PATH'])
    env.pop('SSH_AUTH_SOCK', None)
    return env


def checked_spend(account):
    info = json.loads(subprocess.check_output(['runpodctl', 'me'], env=environment(account), text=True))
    return {k: info[k] for k in ('clientBalance', 'currentSpendPerHr', 'spendLimit')}


def create(worker, spec):
    OPS.mkdir(exist_ok=True)
    receipt = OPS / f'{worker}.json'
    with (OPS / 'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            return json.loads(receipt.read_text())
        pending = OPS / f'{worker}.pending.json'
        name = 'gemma-half-'+worker.lower()+'-20260908'
        fleet = inventory(keys())
        if pending.exists() or any(p['name'] == name for v in fleet.values() for p in v['pods']):
            raise RuntimeError('Existing/ambiguous intent; reconcile before retry: '+worker)
        cost = 3.49 if spec['model'] == '12b' else 4.59
        info = checked_spend(spec['account'])
        # currentSpendPerHr includes storage; API pod sums protect against lag.
        spend = max(info['currentSpendPerHr'], sum(p['costPerHr'] for p in fleet[spec['account']]['pods']))
        reserve = 18.70 if spec['account'] == 'A3' and not any(
            p['name'] == 'glm-8192-a3-glm-1c-control-20260908' for p in fleet['A3']['pods']) else 0
        if spend+cost+0.10+reserve > min(80, info['spendLimit']):
            write(OPS / f'{worker}.waiting-budget.json', dict(spend=spend, proposed=cost,
                reserve=reserve, account=spec['account'], checked=time.time()))
            print('WAITING_BUDGET', worker, flush=True)
            return None
        bind(pending, dict(worker=worker, name=name, account=spec['account'], hourly=cost,
                          provider=info, reserved_glm=reserve, created=time.time()))
        env = environment(spec['account'])
        log = OPS / f'{worker}.create.log'
        with log.open('w') as stream:
            result = subprocess.run([str(SKILL/'create-pod-cuda.sh'), name,
                'NVIDIA H100 80GB HBM3' if spec['model'] == '12b' else 'NVIDIA H200',
                '12.8,12.9,13.0,13.1', 'SECURE', 'runpod-torch-v280', '1',
                '300' if spec['model'] == '12b' else '500'],
                env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=700)
        match = re.search(r'Pod ID:\s+(\w+)', log.read_text())
        if not match:
            raise RuntimeError(f'Unconfirmed allocation {worker}; inspect {log}; rc={result.returncode}')
        pod = dict(worker=worker, account=spec['account'], model=spec['model'],
            name=name, pod_id=match[1], hourly=cost, gpu='H100' if spec['model']=='12b' else 'H200',
            disk_gb=300 if spec['model']=='12b' else 500, root='/workspace/gemma-halfpct/'+worker,
            log='/workspace/gemma-halfpct-worker.log', created_at=time.time())
        bind(receipt, pod)
        # New empty allocation only: inject our existing monitor key if needed.
        resolved = subprocess.check_output(['python3', str(SKILL/'_resolve_ssh.py'),
            pod['pod_id'], '--quiet'], env=env, text=True, timeout=40).split()
        if len(resolved) == 2:
            pod.update(ip=resolved[0], port=int(resolved[1]))
            check = subprocess.run(connection(pod)+['source /etc/rp_environment; printenv RUNPOD_POD_ID'],
                                   capture_output=True, text=True, timeout=20)
        else:
            check = None
        if check is None or check.returncode or check.stdout.strip() != pod['pod_id']:
            subprocess.run(['runpodctl', 'pod', 'update', pod['pod_id'], '--env', json.dumps(
                {'PUBLIC_KEY':Path('/root/.ssh/id_ed25519.pub').read_text().strip()})],
                env=env, check=True, capture_output=True, timeout=60)
        write(receipt, pod)
        print('ALLOCATED', worker, pod['pod_id'], cost, flush=True)
        return pod


def provision(worker, pod, plan):
    env = environment(pod['account'])
    receipt = OPS / f'{worker}.json'
    if (OPS / f'{worker}.launched.json').exists():
        print('ALREADY_LAUNCHED', worker, flush=True)
        return
    actual = [p for p in inventory(keys())[pod['account']]['pods'] if p['id']==pod['pod_id']]
    assert len(actual)==1 and actual[0]['name']==pod['name']
    for attempt in range(40):
        r = subprocess.run(['python3', str(SKILL/'_resolve_ssh.py'), pod['pod_id'], '--quiet'],
                           env=env, text=True, capture_output=True, timeout=30)
        if len(r.stdout.split())==2:
            ip, port = r.stdout.split(); pod.update(ip=ip, port=int(port))
            r = subprocess.run(connection(pod)+['source /etc/rp_environment; printenv RUNPOD_POD_ID'],
                               capture_output=True,text=True,timeout=15)
            if not r.returncode and r.stdout.strip()==pod['pod_id']:
                break
        time.sleep(5)
    else:
        raise RuntimeError('No authenticated SSH: '+worker)
    write(receipt,pod)
    with (OPS/'allocation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(['python3',str(SKILL/'_ssh_alias.py'),'add',pod['name'],pod['pod_id'],
                        pod['ip'],str(pod['port'])],env=env,check=True,capture_output=True)
    with (OPS/'catalog.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        catalog=BASE/'PRODUCTION_PODS.json'
        current=json.loads(catalog.read_text()) if catalog.exists() else {}
        if worker in current:assert current[worker]['pod_id']==pod['pod_id']
        current[worker]=pod;write(catalog,current)
    with (OPS/f'{worker}.preflight.log').open('w') as stream:
        r = subprocess.run([str(SKILL/'pod-preflight.sh'),pod['pod_id'],'12.6'],env=env,
                           stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    assert not r.returncode,'Preflight failed: '+worker
    api = HfApi()
    repo = f"arcadia-impact/scimt-dispatch-gemma-{pod['model']}-aft-grid-v2"
    for job in plan['workers'][worker]['jobs']:
        try:
            exists = list(api.list_repo_tree(repo,path_in_repo=f"followups/{plan['version']}/{job['id']}"))
        except EntryNotFoundError:
            exists = []
        assert not exists, 'Output exists; preserve/reconcile instead of duplicate launch'
    root = pod['root']
    subprocess.run(connection(pod)+['test ! -e /workspace/HALFPCT_LAUNCHED.json; '
        'mkdir -p /workspace/scimt /workspace/gemma-halfpct-prepared '+root+'/data-receipts'],check=True,timeout=30)
    files = [(BASE/n,'/workspace/'+n) for n in ('base-code.tar.gz','code-overlay.tar.gz',
                                             'deployment-code.tar.gz','prepared-inputs.tar.gz')]
    files.append((BASE/f"publish-{pod['model']}/data-receipts/shared-data.json",root+'/data-receipts/shared-data.json'))
    assignment = OPS/f'{worker}.assignment.json'
    bind(assignment,dict(worker=worker,jobs=plan['workers'][worker]['jobs'],pod_id=pod['pod_id'],plan_sha256=sha(BASE/'plan.json')))
    files.append((assignment,root+'/MONITOR_ASSIGNMENT.json'))
    for local,remote in files:
        subprocess.run(connection(pod,True)+[str(local),'root@'+pod['ip']+':'+remote],check=True,timeout=300)
    token=get_token(); assert token
    command=('set -e; set -a; source /etc/rp_environment; set +a; read -r HF_TOKEN; export HF_TOKEN; '
        'export PATH=/root/.local/bin:$PATH; '
        'tar -xzf /workspace/base-code.tar.gz -C /workspace/scimt; '
        'tar -xzf /workspace/code-overlay.tar.gz -C /workspace/scimt; '
        'tar -xzf /workspace/deployment-code.tar.gz -C /workspace/scimt; '
        'tar -xzf /workspace/prepared-inputs.tar.gz -C /workspace/gemma-halfpct-prepared; '
        'test ! -e /workspace/HALFPCT_LAUNCHED.json; cp '+root+'/MONITOR_ASSIGNMENT.json /workspace/HALFPCT_LAUNCHED.json; '
        'tmux new-session -d -s gemma-halfpct "bash /workspace/scimt/experiments/prior_coins/dispatch_final_v1/run_gemma_halfpct_sharded.sh '
        +worker+' >/workspace/gemma-halfpct-worker.log 2>&1"; echo LAUNCHED')
    r=subprocess.run(connection(pod)+[command],input=token+'\n',capture_output=True,text=True,timeout=120)
    assert not r.returncode,r.stderr
    bind(OPS/f'{worker}.launched.json',dict(pod=pod,launched_at=time.time(),output=r.stdout))
    print('LAUNCHED',worker,pod['pod_id'],flush=True)
    with (OPS/f'{worker}.status.log').open('w') as stream:
        subprocess.run([str(SKILL/'pod-status.sh'),pod['pod_id']],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',nargs='+',required=True);a=p.parse_args()
    plan=json.loads((BASE/'plan.json').read_text());validate(plan,BASE/'data')
    for name,digest in json.loads((BASE/'BUNDLES.json').read_text()).items():
        assert sha(BASE/name)==digest
    def one(worker):
        try:
            pod=create(worker,plan['workers'][worker])
            if pod:provision(worker,pod,plan)
        except Exception as error:
            print('FAILED',worker,type(error).__name__,str(error),flush=True)
            return False
        return True
    with ThreadPoolExecutor(3) as pool:
        results=list(pool.map(one,a.workers))
    if not all(results):raise SystemExit(1)


if __name__=='__main__':main()
