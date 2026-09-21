"""Transfer and launch already-created/preflighted Gemma worker pods."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from huggingface_hub import get_token

BASE=Path('artifacts/aft_grid_8192_balanced_v2')


def provision(pair):
    worker,p=pair
    options=['-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
             '-i','/root/.ssh/id_ed25519']
    ssh=['env','-u','SSH_AUTH_SOCK','ssh',*options,'-p',str(p['port']),'root@'+p['ip']]
    identity=subprocess.run(ssh+['source /etc/rp_environment; printenv RUNPOD_POD_ID'],capture_output=True,text=True,timeout=30)
    if identity.returncode or identity.stdout.strip()!=p['pod_id']:
        raise RuntimeError(f'{worker}: remote pod identity not verified; refusing writes')
    live=subprocess.run(ssh+['pgrep -af "[r]un_gemma_grid|[g]emma_grid_run|[b]ash.*pod/setup.sh"'],capture_output=True,text=True,timeout=30)
    if live.returncode==0:
        raise RuntimeError(f'{worker}: existing launch processes; inspect before re-provisioning')
    if live.returncode!=1:
        raise RuntimeError(f'{worker}: SSH check failed: {live.stderr}')
    for name in ('production-code.tar.gz','production-input.tar.gz'):
        subprocess.run(['env','-u','SSH_AUTH_SOCK','scp',*options,'-P',str(p['port']),str(BASE/name),
                        'root@'+p['ip']+':/workspace/'+name],check=True,timeout=240)
    command=('set -e; read -r HF_TOKEN; export HF_TOKEN; '
        'mkdir -p /workspace/scimt /workspace/grid-input; '
        'tar -xzf /workspace/production-code.tar.gz -C /workspace/scimt; '
        'tar -xzf /workspace/production-input.tar.gz -C /workspace/grid-input; '
        'cd /workspace/scimt; nohup bash experiments/prior_coins/dispatch_final_v1/run_gemma_grid.sh '
        +worker+' >/workspace/gemma-grid-worker.log 2>&1 </dev/null & echo LAUNCHED:$!')
    token=get_token()
    if not token:raise RuntimeError('Missing HF credential')
    r=subprocess.run(ssh+[command],input=token+'\n',text=True,capture_output=True,timeout=180)
    if r.returncode:raise RuntimeError(f'{worker}: launch failed: {r.stderr}')
    print(worker,r.stdout,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='append',required=True);a=p.parse_args()
    pods=json.loads((BASE/'PRODUCTION_PODS.json').read_text())
    with ThreadPoolExecutor(4) as pool:list(pool.map(provision,[(w,pods[w]) for w in a.worker]))
