"""Provision one already-created single-H200 pod for a charter-dominant worker.

Creation is deliberate and separate (RunPod MCP / REST: 1x NVIDIA H200 SECURE,
500 GB container disk, image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404,
PUBLIC_KEY = /workspace/.ssh/id_ed25519.pub, name containing "-keep"). This script
never creates, stops or deletes a pod. It:

  1. resolves the pod's SSH endpoint from the RunPod REST v2 API (never cached),
  2. verifies the remote RUNPOD_POD_ID matches and no worker is already running,
  3. ships the committed repo snapshot (git archive HEAD) and the prepared release
     (plan.json, READY.json, data/, data-receipts/) as tarballs,
  4. starts run_gemma_charter_dominant.sh <worker> in tmux, passing HF_TOKEN over
     stdin (never argv/env in a command line).

Preconditions: `publish-data` has been run once from this box for the release
(so <release>/data-receipts/shared-data.json exists), and the plan validates.

    python -m experiments.prior_coins.dispatch_final_v1.ops.provision_charter_dominant \
        --pod-id <id> --worker charter-27b-cd --release artifacts/aft_charter_dominant_v1/release
"""
import argparse
import json
import os
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from experiments.prior_coins.dispatch_final_v1 import gemma_charter_dominant as CD
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind, sha

REPO = Path(__file__).resolve().parents[4]
KEY = Path('/workspace/.ssh/id_ed25519')
IMAGE = 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'


def pod_info(pod_id):
    # REST v2 (the v1 host answers 403 for this key). Fields used: status,
    # gpu.{count,id}, cost, runtime.ports[type=tcp, private=22] -> ip/public.
    req = urllib.request.Request(f'https://api.runpod.io/v2/pods/{pod_id}',
                                 headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'],
                                          'User-Agent': 'curl/8.5.0'})  # the default urllib UA is rejected with 403
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def ssh_target(info):
    for port in ((info.get('runtime') or {}).get('ports') or []):
        if port.get('type') == 'tcp' and port.get('private') == 22 and port.get('ip'):
            return port['ip'], int(port['public'])
    raise RuntimeError('Pod has no public SSH port yet')


def ssh(ip, port, *cmd, input=None, timeout=120, scp=None):
    opts = ['-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
            '-i', str(KEY)]
    env = {k: v for k, v in os.environ.items() if k != 'SSH_AUTH_SOCK'}
    if scp:
        return subprocess.run(['scp', *opts, '-P', str(port), str(scp[0]), f'root@{ip}:{scp[1]}'],
                              env=env, check=True, timeout=timeout)
    return subprocess.run(['ssh', *opts, '-p', str(port), f'root@{ip}', *cmd], env=env,
                          input=input, capture_output=True, text=True, timeout=timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pod-id', required=True)
    p.add_argument('--worker', required=True, choices=[CD.worker_name(a) for a in CD.ARMS])
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--execute', action='store_true', help='without it: verify only, ship nothing')
    a = p.parse_args()
    release = a.release.resolve()
    plan = json.loads((release / 'plan.json').read_text())
    CD.validate(plan, release / 'data')
    if not (release / 'data-receipts/shared-data.json').exists():
        raise RuntimeError('Run publish-data first; the worker refuses to start without the receipt')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if plan['base_commit'] != head:
        changed = subprocess.check_output(['git', 'diff', '--name-only', plan['base_commit'], head, '--',
                                           'experiments', 'src', 'requirements'], cwd=REPO, text=True).split()
        outside_ops = [f for f in changed if not f.startswith('experiments/prior_coins/dispatch_final_v1/ops/')]
        if outside_ops:
            raise RuntimeError('HEAD differs from the plan commit in runtime files: ' + ', '.join(outside_ops))
    if subprocess.check_output(['git', 'status', '--porcelain', '--', 'experiments', 'src', 'requirements'],
                               cwd=REPO, text=True).strip():
        raise RuntimeError('Uncommitted changes under experiments/ src/ requirements/; commit or stash')
    info = pod_info(a.pod_id)
    if '-keep' not in info['name']:
        raise RuntimeError(f"Pod name {info['name']!r} lacks -keep; both sweepers would stop it mid-run")
    if info.get('status') != 'RUNNING':
        raise RuntimeError(f"Pod is {info.get('status')}, not RUNNING")
    gpu = (info.get('gpu') or {}).get('id', '')
    if 'H200' not in gpu or int((info.get('gpu') or {}).get('count', 0)) != 1:
        raise RuntimeError(f'Expected 1x H200, got {gpu} x {(info.get("gpu") or {}).get("count")}')
    ip, port = ssh_target(info)
    for _ in range(30):
        r = ssh(ip, port, 'source /etc/rp_environment; printenv RUNPOD_POD_ID', timeout=20)
        if r.returncode == 0 and r.stdout.strip() == a.pod_id:
            break
        time.sleep(10)
    else:
        raise RuntimeError('No authenticated SSH with matching pod identity')
    live = ssh(ip, port, 'pgrep -af "[g]emma_charter_dominant|[b]ash.*pod/setup.sh" || true')
    if live.stdout.strip():
        raise RuntimeError('A worker or setup is already running on this pod:\n' + live.stdout)
    receipt = release / 'deployment' / f'{a.worker}.json'
    if receipt.exists():
        raise RuntimeError(f'{receipt} exists; this worker was already provisioned somewhere')
    print(json.dumps(dict(pod=a.pod_id, name=info['name'], ip=ip, port=port, gpu=gpu,
                          cost=info.get('cost'), worker=a.worker, execute=a.execute)), flush=True)
    if not a.execute:
        return
    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp) / 'code.tar.gz'
        with code.open('wb') as f:
            subprocess.run(['git', 'archive', '--format=tar.gz', 'HEAD'], cwd=REPO, stdout=f, check=True)
        inputs = Path(tmp) / 'input.tar.gz'
        with tarfile.open(inputs, 'w:gz') as t:
            for name in ('plan.json', 'READY.json', 'DATA_AUDIT.json', 'data', 'data-receipts'):
                t.add(release / name, arcname=name)
        ssh(ip, port, 'mkdir -p /workspace/scimt /workspace/cd-input && test ! -e /workspace/CD_LAUNCHED.json',
            timeout=30).check_returncode()
        ssh(ip, port, scp=(code, '/workspace/code.tar.gz'), timeout=600)
        ssh(ip, port, scp=(inputs, '/workspace/input.tar.gz'), timeout=600)
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN missing from the environment (source /workspace/.env)')
    cmd = ('set -e; read -r HF_TOKEN; export HF_TOKEN; '
           'tar -xzf /workspace/code.tar.gz -C /workspace/scimt; '
           'tar -xzf /workspace/input.tar.gz -C /workspace/cd-input; '
           'command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh); export PATH=/root/.local/bin:$PATH; '
           'echo \'{"worker":"' + a.worker + '"}\' > /workspace/CD_LAUNCHED.json; '
           'tmux new-session -d -s cd "bash /workspace/scimt/experiments/prior_coins/dispatch_final_v1/'
           'run_gemma_charter_dominant.sh ' + a.worker + ' >/workspace/cd-worker.log 2>&1"; echo LAUNCHED')
    r = ssh(ip, port, cmd, input=token + '\n', timeout=180)
    if r.returncode or 'LAUNCHED' not in r.stdout:
        raise RuntimeError(f'Launch failed: {r.stderr}\n{r.stdout}')
    bind(receipt, dict(pod_id=a.pod_id, name=info['name'], ip=ip, port=port, worker=a.worker,
                       plan_sha256=sha(release / 'plan.json'), launched_at=time.time(),
                       log='/workspace/cd-worker.log', root=f'/workspace/gemma-cd/{a.worker}',
                       status_file=f'/workspace/gemma-cd/{a.worker}/STATUS.json'))
    print('LAUNCHED', a.worker, a.pod_id, flush=True)


if __name__ == '__main__':
    main()
