"""Withdraw one unstarted source queue, record transfer, launch its new owner."""
import argparse
import json
from pathlib import Path
import subprocess

from huggingface_hub import HfApi, get_token
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import bind, sha

BASE=Path('artifacts/aft_grid_8192_balanced_v2')


def connection(pod,scp=False):
    options=['-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10',
             '-o','StrictHostKeyChecking=accept-new','-i','/root/.ssh/id_ed25519']
    if pod.get('ssh_proxy'): options+=['-o','ProxyCommand='+pod['ssh_proxy']]
    if scp:return ['env','-u','SSH_AUTH_SOCK','scp',*options,'-P',str(pod['port'])]
    return ['env','-u','SSH_AUTH_SOCK','ssh',*options,'-p',str(pod['port']),'root@'+pod['ip']]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',required=True);a=parser.parse_args()
    pods=json.loads((BASE/'PRODUCTION_PODS.json').read_text())
    dest=pods[a.worker];source_name=dest['logical_worker'];source=pods[source_name]
    assert dest['phase']=='repair' and dest['gpu']=='H200'
    assert (BASE/'deploy'/f'{a.worker}.ready.json').exists()
    plan_path=BASE/'repair-plan-52cells.json';plan=json.loads(plan_path.read_text())
    jobs=plan['workers'][source_name]['jobs'];assert len(jobs)==4
    for pod in (source,dest):
        actual=subprocess.check_output(connection(pod)+['source /etc/rp_environment; printenv RUNPOD_POD_ID'],text=True,timeout=30).strip()
        assert actual==pod['pod_id']
    # Refuse duplicate launch/reprovision even if the original withdrawal succeeded.
    check=subprocess.run(connection(dest)+['test ! -e /workspace/REPAIR_LAUNCHED.json'],timeout=30)
    assert check.returncode==0,'Already launched; inspect instead of repeating'
    api=HfApi();repo='arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2'
    for job in jobs:
        try:
            existing=list(api.list_repo_tree(repo,path_in_repo=f"followups/{plan['version']}/{job['id']}",recursive=True))
        except Exception as exc:
            if getattr(getattr(exc,'response',None),'status_code',None)!=404:raise
            existing=[]
        assert not existing,'Existing result path; reconcile instead of duplicate training'
    subprocess.run(connection(source,True)+[
        'experiments/dispatch/dispatch_final_v1/ops/transfer_gemma_repair.py',
        'root@'+source['ip']+':/workspace/transfer_gemma_repair.py'],check=True,timeout=60)
    result=subprocess.run(connection(source)+[
        f'python3 /workspace/transfer_gemma_repair.py --worker {source_name} --destination {a.worker} --execute'],
        text=True,capture_output=True,timeout=40)
    assert result.returncode==0,result.stderr
    transfer=json.loads(result.stdout)
    assert transfer['status']=='transferred_unstarted' and transfer['jobs']==jobs
    assert transfer['plan_sha256']==sha(plan_path)
    transfer.update(source_pod=source['pod_id'],destination_pod=dest['pod_id'])
    receipt=BASE/'reshard'/f'{a.worker}-transfer.json';bind(receipt,transfer)
    # Original plan remains byte-identical. Physical account/pod ownership is
    # recorded separately, so all job IDs and HF scientific namespaces stay fixed.
    commands='mkdir -p /workspace/scimt /workspace/grid-input /workspace/grid-repair-input '+dest['root']+'/data-receipts'
    subprocess.run(connection(dest)+[commands],check=True,timeout=30)
    files=[(BASE/'production-code.tar.gz','/workspace/production-code.tar.gz'),
       (BASE/'production-input.tar.gz','/workspace/production-input.tar.gz'),
       (BASE/'repair-code-overlay.tar.gz','/workspace/grid-repair-input/code.tar.gz'),
       (plan_path,'/workspace/grid-repair-input/repair-plan-52cells.json'),
       (BASE/'repair-publish-27b/data-receipts/shared-data.json',dest['root']+'/data-receipts/shared-data.json'),
       (receipt,dest['root']+'/TRANSFER_IN.json'),
       (Path('experiments/dispatch/dispatch_final_v1/run_gemma_repair_direct.sh'),'/workspace/run_gemma_repair_direct.sh')]
    for local,remote in files:
        subprocess.run(connection(dest,True)+[str(local),'root@'+dest['ip']+':'+remote],check=True,timeout=240)
    token=get_token();assert token
    command=('set -e; read -r HF_TOKEN; export HF_TOKEN; '
      'tar -xzf /workspace/production-code.tar.gz -C /workspace/scimt; '
      'tar -xzf /workspace/grid-repair-input/code.tar.gz -C /workspace/scimt; '
      'tar -xzf /workspace/production-input.tar.gz -C /workspace/grid-input; '
      'test ! -e /workspace/REPAIR_LAUNCHED.json; '
      'cp '+dest['root']+'/TRANSFER_IN.json /workspace/REPAIR_LAUNCHED.json; '
      'nohup bash /workspace/run_gemma_repair_direct.sh '+source_name+
      ' >/workspace/gemma-repair-worker.log 2>&1 </dev/null & echo LAUNCHED:$!')
    result=subprocess.run(connection(dest)+[command],input=token+'\n',text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stderr
    bind(BASE/'reshard'/f'{a.worker}-launch.json',dict(worker=a.worker,pod_id=dest['pod_id'],
        logical_worker=source_name,root=dest['root'],log=dest['log'],launch_output=result.stdout.strip(),transfer=str(receipt)))
    print(a.worker,result.stdout,flush=True)


if __name__=='__main__':main()
