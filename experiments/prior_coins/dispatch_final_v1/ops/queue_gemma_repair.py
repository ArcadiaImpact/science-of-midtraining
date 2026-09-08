"""Install the authorized #1c continuation on existing owned workers only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from huggingface_hub import get_token
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind

BASE=Path('artifacts/aft_grid_8192_balanced_v2')


def install(pair):
    worker,pod=pair;model=worker.split('-')[1]
    options=['-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-i','/root/.ssh/id_ed25519']
    ssh=['env','-u','SSH_AUTH_SOCK','ssh',*options,'-p',str(pod['port']),'root@'+pod['ip']]
    identity=subprocess.check_output(ssh+['source /etc/rp_environment; printenv RUNPOD_POD_ID'],text=True,timeout=30).strip()
    if identity!=pod['pod_id']:raise RuntimeError('Pod ownership mismatch')
    check=subprocess.run(ssh+['test ! -e /workspace/gemma-grid/'+worker+'/CONTINUATION_PENDING.json'],timeout=30)
    if check.returncode:raise RuntimeError(worker+': already registered; reconcile instead of duplicate launch')
    subprocess.run(ssh+['mkdir -p /workspace/grid-repair-input /workspace/gemma-grid-repair/'+worker+'/data-receipts; test -f /workspace/production-code.tar.gz'],check=True,timeout=30)
    scp=['env','-u','SSH_AUTH_SOCK','scp',*options,'-P',str(pod['port'])]
    for source,dest in [(BASE/'repair-code-overlay.tar.gz','/workspace/grid-repair-input/code.tar.gz'),
                        (BASE/'repair-plan-52cells.json','/workspace/grid-repair-input/repair-plan-52cells.json'),
                        (BASE/f'repair-publish-{model}/data-receipts/shared-data.json',
                         '/workspace/gemma-grid-repair/'+worker+'/data-receipts/shared-data.json')]:
        subprocess.run(scp+[str(source),'root@'+pod['ip']+':'+dest],check=True,timeout=90)
    # Isolated code tree: no changed Python files under the active run's checkout.
    setup=('set -e; mkdir /workspace/scimt-1c; '
           'tar -xzf /workspace/production-code.tar.gz -C /workspace/scimt-1c; '
           'tar -xzf /workspace/grid-repair-input/code.tar.gz -C /workspace/scimt-1c; '
           'cd /workspace/scimt-1c; '
           'PYTHONPATH=/workspace/scimt-1c:/workspace/scimt-1c/src python3 -m '
           'experiments.prior_coins.dispatch_final_v1.gemma_grid_run worker '
           '--worker '+worker+' --plan /workspace/grid-repair-input/repair-plan-52cells.json '
           '--data /workspace/grid-input/data-validated --root /workspace/gemma-grid-repair/'+worker+' '
           '--publish-repo arcadia-impact/scimt-dispatch-gemma-'+model+'-aft-grid-v2 >/workspace/gemma-repair-preflight.log')
    subprocess.run(ssh+[setup],check=True,timeout=120)
    token=get_token()
    if not token:raise RuntimeError('Missing HF token')
    launch=('set -e; read -r HF_TOKEN; export HF_TOKEN; '
            'nohup bash /workspace/scimt-1c/experiments/prior_coins/dispatch_final_v1/run_gemma_repair.sh '
            +worker+' >/workspace/gemma-repair-worker.log 2>&1 </dev/null & echo $!')
    r=subprocess.run(ssh+[launch],input=token+'\n',text=True,capture_output=True,timeout=30,check=True)
    result=dict(worker=worker,pod_id=identity,pid=int(r.stdout.strip()),root='/workspace/gemma-grid-repair/'+worker)
    bind(BASE/'repair-launches'/f'{worker}.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='append');a=p.parse_args()
    pods=json.loads((BASE/'PRODUCTION_PODS.json').read_text())
    with ThreadPoolExecutor(4) as pool:list(pool.map(install,[(w,r) for w,r in pods.items() if not a.worker or w in a.worker]))
