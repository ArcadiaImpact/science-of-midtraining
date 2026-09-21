"""One-at-a-time coordinator for the explicitly authorized GLM 5% retirement.

Uses the RunPod skill for all deletion; never allocates or moves other jobs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import get_token
from experiments.dispatch.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import keys, inventory
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import write

DIRECTORY=Path('artifacts/aft_size_mixture_v1/paused_5pct')
ORDER=['A2-charter','A2-coin','A2-control','A3-charter','A3-coin','A3-control']
SKILL=Path('/root/.codex/skills/runpod-spinup')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('worker',choices=ORDER)
    p.add_argument('--phase',choices=['launch','verify-cleanup'],required=True)
    p.add_argument('--cli-dir',type=Path,required=True)
    a=p.parse_args()
    account,arm=a.worker.split('-')
    for previous in ORDER[:ORDER.index(a.worker)]:
        assert json.loads((DIRECTORY/f'{previous}-cleanup.json').read_text())['confirmed_absent']
    rows=[line.split('\t') for line in Path('experiments/dispatch/dispatch_final_v1/ops/handrun_units.tsv').read_text().splitlines()
          if line.startswith(f'glm-aft81920/{account}/{arm}\t')]
    assert len(rows)==1
    row=rows[0];pod,host=row[2],row[3]
    credentials=keys()
    assert any(v['id']==pod and v['name']==f'glm-aft81920-{account.lower()}-{arm}-20260907'
               for v in inventory(credentials)[account]['pods'])
    opts=['-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-i','/root/.ssh/id_ed25519']
    ssh=['env','-u','SSH_AUTH_SOCK','ssh',*opts,host]
    if a.phase=='launch':
        subprocess.run(['env','-u','SSH_AUTH_SOCK','scp',*opts,
          'experiments/dispatch/dispatch_final_v1/ops/pause_glm_5pct.py',host+':/workspace/pause_glm_5pct.py'],check=True)
        command=f'/usr/local/bin/python /workspace/pause_glm_5pct.py --account {account} --arm {arm} --pod-id {pod}'
        subprocess.run(ssh+[command],check=True)
        # Duplicate launch refusal: exact script matching excludes this SSH shell.
        check="/usr/local/bin/python -c \"import sys; sys.path.insert(0,'/workspace'); from pause_glm_5pct import processes; assert not any('/workspace/pause_glm_5pct.py' in p['argv'] for p in processes())\""
        subprocess.run(ssh+[check],check=True)
        launch=('set -e; read -r HF_TOKEN; export HF_TOKEN HF_HOME=/workspace/hf-final-v1 '
          'FINAL_V1_PROFILE=glm45_air_190m SCIMT_APPLY_LOADER_PATCH=0 NCCL_NVLS_ENABLE=0 '
          'CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/workspace/scimt:/workspace/scimt/src; '
          f'nohup {command} --execute > /workspace/pause-5pct.log 2>&1 < /dev/null & echo ARCHIVE_PID=$!')
        token=get_token();assert token
        result=subprocess.run(ssh+[launch],input=token+'\n',text=True,capture_output=True,timeout=30)
        print(result.stdout);print(result.stderr);assert result.returncode==0
        return
    receipt=DIRECTORY/f'{a.worker}-verified.json'
    subprocess.run(['uv','run','--with','huggingface_hub','--with','requests','python','-m',
       'experiments.dispatch.dispatch_final_v1.ops.verify_paused_glm',
       '--account',account,'--arm',arm,'--out',str(receipt)],check=True)
    r=json.loads(receipt.read_text())
    assert r['pod']['id']==pod and time.time()-r['verified_at']<600
    env={**os.environ,'RUNPOD_API_KEY':credentials[account],'PATH':str(a.cli_dir)+':'+os.environ['PATH']}
    meta=json.loads(subprocess.check_output(['runpodctl','pod','get',pod,'-o','json'],env=env))
    assert meta['id']==pod and meta['name']==r['pod']['name']
    created=datetime.strptime(meta['createdAt'][:19].replace('T',' '),'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    subprocess.run([str(SKILL/'cleanup-pod.sh'),pod],env=env,check=True)
    subprocess.run([str(SKILL/'cleanup-pod.sh'),pod,'--yes'],env=env,check=True)
    assert not any(v['id']==pod for v in inventory(credentials)[account]['pods'])
    now=time.time()
    cleanup={'pod':r['pod'],'created_at':meta['createdAt'],'deleted_at':now,
       'approx_gpu_spend_usd':round((now-created)/3600*float(meta['costPerHr']),2),
       'verified_receipt':receipt.name,'remote_commit':r['archive']['commit'],'confirmed_absent':True}
    write(DIRECTORY/f'{a.worker}-cleanup.json',cleanup)
    subprocess.run([str(SKILL/'pod-status.sh'),pod],env=env,check=False)
    print(json.dumps(cleanup),flush=True)


if __name__=='__main__':
    main()
