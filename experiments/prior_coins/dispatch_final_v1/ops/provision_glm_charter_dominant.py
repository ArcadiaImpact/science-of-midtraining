"""Provision one already-created 4xH200 pod for a GLM charter-dominant worker.

Same contract as provision_charter_dominant.py (Gemma): resolves SSH from the RunPod API,
verifies pod identity and hardware, refuses if a worker is already running, ships the
committed repo snapshot + the prepared release, starts setup.sh in tmux with HF_TOKEN over
stdin. Never creates, stops or deletes pods.

    python -m experiments.prior_coins.dispatch_final_v1.ops.provision_glm_charter_dominant \
        --pod-id <id> --worker glm-cd-charter-80 --release artifacts/aft_charter_dominant_v1/glm_release --execute
"""
import argparse
import json
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1.prepare import validate
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind, sha
from experiments.prior_coins.dispatch_final_v1.ops.create_glm_pod import api

REPO = Path(__file__).resolve().parents[4]
KEY = Path('/workspace/.ssh/id_ed25519')


def ssh_target(info):
    for port in ((info.get('runtime') or {}).get('ports') or []):
        if port.get('type') == 'tcp' and port.get('private') == 22 and port.get('ip'):
            return port['ip'], int(port['public'])
    raise RuntimeError('Pod has no public SSH port yet')


def ssh(ip, port, *cmd, input=None, timeout=120, scp=None):
    opts = ['-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-i', str(KEY)]
    env = {k: v for k, v in os.environ.items() if k != 'SSH_AUTH_SOCK'}
    if scp:
        return subprocess.run(['scp', *opts, '-P', str(port), str(scp[0]), f'root@{ip}:{scp[1]}'], env=env, check=True, timeout=timeout)
    return subprocess.run(['ssh', *opts, '-p', str(port), f'root@{ip}', *cmd], env=env, input=input,
                          capture_output=True, text=True, timeout=timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pod-id', required=True)
    p.add_argument('--worker', required=True, choices=sorted(C.PLACEMENT))
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    release = a.release.resolve()
    plan = json.loads((release / 'plan.json').read_text())
    validate(plan, release / 'data')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--', 'experiments', 'src', 'requirements'], cwd=REPO, text=True).strip():
        raise RuntimeError('Uncommitted changes under experiments/ src/ requirements/; commit first')
    info = api('GET', f'/pods/{a.pod_id}')
    if '-keep' not in info['name']:
        raise RuntimeError(f"Pod name {info['name']!r} lacks -keep")
    if info.get('status') != 'RUNNING':
        raise RuntimeError(f"Pod is {info.get('status')}, not RUNNING")
    gpu = info.get('gpu') or {}
    if 'H200' not in gpu.get('id', '') or int(gpu.get('count', 0)) != 4:
        raise RuntimeError(f'Expected 4x H200, got {gpu}')
    ip, port = ssh_target(info)
    for _ in range(30):
        r = ssh(ip, port, 'source /etc/rp_environment; printenv RUNPOD_POD_ID', timeout=20)
        if r.returncode == 0 and r.stdout.strip() == a.pod_id:
            break
        time.sleep(10)
    else:
        raise RuntimeError('No authenticated SSH with matching pod identity')
    mem = ssh(ip, port, "free -g | awk '/Mem:/{print $2}'; nproc; df -BG /workspace | tail -1 | awk '{print $4}'", timeout=20).stdout.split()
    if len(mem) < 3 or int(mem[0]) < 1000 or int(mem[2].rstrip('G')) < 1400:
        raise RuntimeError(f'Host does not meet the floor (RAM GB, vCPU, free GB): {mem}')
    live = ssh(ip, port, 'pgrep -af "glm_aft_charter_dominant|bash.*pod/setup.sh" | grep -v pgrep || true')
    if live.stdout.strip():
        raise RuntimeError('A worker or setup is already running on this pod:\n' + live.stdout)
    receipt = release / 'deployment' / f'{a.worker}.launched.json'
    if receipt.exists():
        raise RuntimeError(f'{receipt} exists; already provisioned')
    print(json.dumps(dict(pod=a.pod_id, name=info['name'], ip=ip, port=port, gpu=gpu, host_ram_gb=mem[0], vcpu=mem[1],
                          free_gb=mem[2], cost=info.get('cost'), worker=a.worker, head=head[:8], execute=a.execute)), flush=True)
    if not a.execute:
        return
    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp) / 'code.tar.gz'
        with code.open('wb') as f:
            subprocess.run(['git', 'archive', '--format=tar.gz', 'HEAD'], cwd=REPO, stdout=f, check=True)
        inputs = Path(tmp) / 'input.tar.gz'
        with tarfile.open(inputs, 'w:gz') as t:
            for name in ('plan.json', 'READY.json', 'PARENTS.json', 'TOKENIZER_AUDIT.json', 'EVAL_INPUTS.json', 'data'):
                t.add(release / name, arcname=name)
        ssh(ip, port, 'mkdir -p /workspace/scimt /workspace/glm-cd-prepared && test ! -e /workspace/GLM_CD_LAUNCHED.json', timeout=30).check_returncode()
        ssh(ip, port, scp=(code, '/workspace/code.tar.gz'), timeout=600)
        ssh(ip, port, scp=(inputs, '/workspace/input.tar.gz'), timeout=600)
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN missing (source /workspace/.env)')
    cmd = ('set -e; read -r HF_TOKEN; export HF_TOKEN; set -a; source /etc/rp_environment; set +a; '
           'tar -xzf /workspace/code.tar.gz -C /workspace/scimt; tar -xzf /workspace/input.tar.gz -C /workspace/glm-cd-prepared; '
           'command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh); export PATH=/root/.local/bin:$PATH; '
           'echo \'{"worker":"' + a.worker + '"}\' > /workspace/GLM_CD_LAUNCHED.json; '
           'tmux new-session -d -s cd "bash /workspace/scimt/experiments/prior_coins/dispatch_final_v1/glm_aft_charter_dominant_v1/setup.sh '
           + a.worker + ' >/workspace/glm-cd-worker.log 2>&1"; echo LAUNCHED')
    r = ssh(ip, port, cmd, input=token + '\n', timeout=180)
    if r.returncode or 'LAUNCHED' not in r.stdout:
        raise RuntimeError(f'Launch failed: {r.stderr}\n{r.stdout}')
    bind(receipt, dict(pod_id=a.pod_id, name=info['name'], ip=ip, port=port, worker=a.worker, code_commit=head,
                       plan_sha256=sha(release / 'plan.json'), launched_at=time.time(),
                       log='/workspace/glm-cd-worker.log', status_file=f'/workspace/glm-cd/{a.worker}/STATUS.json'))
    print('LAUNCHED', a.worker, a.pod_id, flush=True)


if __name__ == '__main__':
    main()
