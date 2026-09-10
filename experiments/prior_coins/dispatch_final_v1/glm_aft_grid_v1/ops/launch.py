"""Box-side launcher for the GLM grid pods (one RunPod account; raw GraphQL for minMemoryInGb).

  python -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.launch plan
  python -m ...ops.launch create    --worker glm-grid-charter [--dry-run]   # spend check, GraphQL create, pod-own add
  python -m ...ops.launch provision --worker glm-grid-charter               # wait ssh (12 min cap), preflight, ship, start
  python -m ...ops.launch bundle                                            # (re)build the three tarballs only

Copied in spirit from ops/launch_glm_repair.py (#1c): 4xH200 SECURE, template
runpod-torch-v280, 2,000 GB container disk, minMemoryInGb 1000, both public
keys in PUBLIC_KEY, Hub token piped to /root/.hf_token under umask 077.
Never stops or deletes a pod.  Creation refuses when the account would exceed
its hourly cap or already runs three GLM grid pods.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.common import (
    DEPLOY, POD_OWN, PREPARED, PROGRESS_GLOB, clean_env, graphql, myself, now, pod_get, receipt_path,
    scp_to, spend_check, ssh_cmd, write_json)
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.prepare import ready_inputs

IMAGE_SUFFIXES = ('.png', '.pdf', '.svg', '.jpg', '.jpeg', '.gif')
DEPLOY_GIT = C.REPO / 'experiments/prior_coins/dispatch_final_v1/ops/glm_deployment_git.py'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bundle(force=False):
    """base-code = tracked files minus images; code-overlay = this package; prepared-inputs = the release."""
    DEPLOY.mkdir(parents=True, exist_ok=True)
    manifest = DEPLOY / 'DEPLOYED_SOURCE.json'
    if manifest.exists() and not force and all((DEPLOY / n).exists() for n in
                                                ('base-code.tar.gz', 'code-overlay.tar.gz', 'prepared-inputs.tar.gz')):
        return json.loads(manifest.read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C.REPO, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=C.REPO, text=True)
    if dirty.strip():
        raise RuntimeError(f'Tracked files modified in the worktree; commit or revert first:\n{dirty[:2000]}')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=C.REPO).decode().split('\0')
    tracked = [t for t in tracked if t and not t.lower().endswith(IMAGE_SUFFIXES)]
    with tarfile.open(DEPLOY / 'base-code.tar.gz', 'w:gz') as tar:
        for rel in tracked:
            tar.add(C.REPO / rel, arcname=rel, recursive=False)
    overlay_root = C.HERE.relative_to(C.REPO)
    with tarfile.open(DEPLOY / 'code-overlay.tar.gz', 'w:gz') as tar:
        for p in sorted(C.HERE.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                tar.add(p, arcname=str(overlay_root / p.relative_to(C.HERE)))
    with tarfile.open(DEPLOY / 'prepared-inputs.tar.gz', 'w:gz') as tar:
        for p in sorted(PREPARED.rglob('*')):
            if p.is_file() and 'deployment' not in p.relative_to(PREPARED).parts:
                tar.add(p, arcname=str(p.relative_to(PREPARED)))
    result = dict(worktree=str(C.REPO), head=head, tracked_files=len(tracked), overlay=str(overlay_root),
                  built=now(), archives={n: dict(sha256=sha256(DEPLOY / n), size=(DEPLOY / n).stat().st_size)
                                         for n in ('base-code.tar.gz', 'code-overlay.tar.gz', 'prepared-inputs.tar.gz')})
    write_json(manifest, result)
    return result


def plan_summary(p):
    lines = [f"version {p['version']}  stage {p['stage']}  parent {p['parent_repo']} @ {p['parent_revision'][:12]}",
             f"recipe: {p['recipe']['steps']} steps, saves {p['recipe']['saves']}, evals {p['recipe']['eval_steps']}, "
             f"global batch {p['recipe']['global_batch']} (micro {p['recipe']['microbatch']} x {p['recipe']['gpus']} ranks), seed {p['recipe']['seed']}",
             f"pod: {C.POD}", f"hub: {p['publishing']['hub_repo']} / {p['publishing']['hub_prefix']}/<profile>/<arm>/<mix> (blobs only)",
             f"large files: mirror {p['publishing']['mirror_root']} ; gcs {p['publishing']['gcs_target']} (pending creds)"]
    for w, spec in p['workers'].items():
        lines.append(f"{w} ({spec['arm']}, pod {spec['pod_name']}): " + ' -> '.join(
            f"{j['mix']}[{j['conflict_rows']}]" for j in spec['jobs']))
    lines.append('datasets: ' + ', '.join(f"{m} {d['sha256'][:10]}" for m, d in p['datasets'].items()))
    lines.append('tokens/row (GLM, measured): ' + ', '.join(f"{m} {t['tokens_per_row']}" for m, t in p['tokens'].items()))
    lines.append(f"pinned sources: {len(p['source_hashes'])} files")
    return '\n'.join(lines)


def create(worker, dry_run):
    p = json.loads((PREPARED / 'plan.json').read_text())
    ready_inputs(PREPARED, p)
    spec = p['workers'][worker]
    name = spec['pod_name']
    me = myself()
    report = spend_check(me)
    if any(pod['name'] == name for pod in me['pods']):
        raise RuntimeError(f'A pod named {name} already exists (any state); inspect, never duplicate')
    if receipt_path(worker).exists() or (DEPLOY / f'{worker}.pending.json').exists():
        raise RuntimeError('Existing/ambiguous allocation receipt; inspect, never duplicate')
    public = (Path.home() / '.ssh/id_ed25519.pub').read_text().strip() + '\n' + \
        (Path.home() / '.runpod/ssh/runpodctl-ssh-key.pub').read_text().strip()
    body = '''mutation { podFindAndDeployOnDemand(input: {
      cloudType: SECURE, gpuCount: 4, gpuTypeId: "NVIDIA H200",
      templateId: "runpod-torch-v280", containerDiskInGb: 2000,
      volumeInGb: 0, minMemoryInGb: 1000,
      allowedCudaVersions: ["12.8","12.9","13.0","13.1"],
      ports: "22/tcp,8888/http", startSsh: true, supportPublicIp: true,
      env: [{key: "PUBLIC_KEY", value: SSH_PUBLIC_VALUE}], name: POD_NAME_VALUE
    }) { id machineId costPerHr } }'''.replace('SSH_PUBLIC_VALUE', json.dumps(public)).replace('POD_NAME_VALUE', json.dumps(name))
    log_dir = C.LOG_ROOT / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{spec['arm']}"
    print(json.dumps(dict(worker=worker, pod_name=name, spend=report), indent=1))
    if dry_run:
        print('DRY RUN: would POST podFindAndDeployOnDemand:\n' + body.replace(json.dumps(public), '"<public keys>"'))
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(DEPLOY / f'{worker}.pending.json', dict(worker=worker, name=name, spend=report, at=now(),
                                                      plan_sha256=p and hashlib.sha256((PREPARED / 'plan.json').read_bytes()).hexdigest()))
    try:
        result = graphql(body).get('podFindAndDeployOnDemand')
    except RuntimeError as error:
        if 'SUPPLY_CONSTRAINT' in str(error):
            # Definitive "nothing was created": clear the intent so a snipe loop can retry.
            (DEPLOY / f'{worker}.pending.json').unlink(missing_ok=True)
            print(f'SUPPLY_CONSTRAINT {worker} {now()}', flush=True)
            raise SystemExit(7)
        if 'INTERNAL_SERVER_ERROR' in str(error):
            # Ambiguous (seen on concurrent creates, #1b 2026-09-07): reconcile by name before deciding.
            time.sleep(20)
            live = [pod for pod in myself()['pods'] if pod['name'] == name and pod['desiredStatus'] != 'EXITED']
            if live:
                result = dict(id=live[0]['id'], costPerHr=live[0]['costPerHr'], machineId=None, adopted_after_error=True)
                print(f'INTERNAL_SERVER_ERROR but pod {name} exists: adopting {result["id"]}', flush=True)
            else:
                (DEPLOY / f'{worker}.pending.json').unlink(missing_ok=True)
                print(f'INTERNAL_SERVER_ERROR {worker} {now()}: no pod created; retryable', flush=True)
                raise SystemExit(8)
        else:
            raise
    if not result or not result.get('id'):
        raise RuntimeError('Unconfirmed allocation: reconcile the pending intent against the console')
    receipt = dict(worker=worker, arm=spec['arm'], name=name, pod_id=result['id'], hourly=result['costPerHr'],
                   machine_id=result.get('machineId'), created_at=now(), created_epoch=time.time(),
                   log_dir=str(log_dir), spend_before=report, pod=C.POD)
    write_json(receipt_path(worker), receipt)
    (DEPLOY / f'{worker}.pending.json').unlink()
    write_json(log_dir / 'pod-create.json', dict(receipt, graphql_result=result))
    own = subprocess.run([str(POD_OWN), 'add', result['id'], '--progress-glob', PROGRESS_GLOB, '--stale-min', '90'],
                         env=clean_env(), capture_output=True, text=True)
    (log_dir / 'pod-own.log').write_text(own.stdout + own.stderr)
    print(f"POD CREATED {result['id']} name={name} ${result['costPerHr']}/h machine={result.get('machineId')} "
          f"pod-own rc={own.returncode} log_dir={log_dir}", flush=True)
    return receipt


def wait_ssh(receipt, minutes=12):
    deadline = time.time() + 60 * minutes
    while time.time() < deadline:
        info = pod_get(receipt['pod_id'])
        ssh = (info or {}).get('ssh') or {}
        if ssh.get('ip') and ssh.get('port'):
            receipt.update(ip=ssh['ip'], port=int(ssh['port']), runtime_status=(info or {}).get('runtimeStatus'))
            write_json(receipt_path(receipt['worker']), receipt)
            return receipt
        time.sleep(10)
    return None


def preflight(receipt):
    script = r'''set -a; source /etc/rp_environment 2>/dev/null; set +a
echo POD_ID=$RUNPOD_POD_ID
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | sed 's/^/GPU /'
echo RAM_GB=$(free -g | awk '/Mem:/{print $2}')
echo CGROUP_MAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo none)
echo DISK_GB=$(df -BG /workspace | awk 'NR==2{print $4}' | tr -d G)
echo PY=$(python3 --version 2>&1)
echo TORCH=$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)
echo GIT=$(command -v git || echo none) RSYNC=$(command -v rsync || echo none)
PYPI=$(curl -sS --max-time 20 https://pypi.org/pypi/nvidia-cudnn-cu12/json | python3 -c 'import json,sys; u=[i for i in json.load(sys.stdin).get("urls",[]) if i["filename"].endswith(".whl")]; print(max(u,key=lambda i:i.get("size",0))["url"])' 2>/dev/null || echo https://files.pythonhosted.org/)
for u in 'https://download.pytorch.org/whl/cu126/torch-2.12.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl' "$PYPI" 'https://huggingface.co/zai-org/GLM-4.5-Air-Base/resolve/main/model-00001-of-00046.safetensors'; do
  s=$(curl -sS -o /dev/null -w '%{speed_download}' --max-time 15 -r '0-300000000' "$u" 2>/dev/null || echo 0); echo "PROBE ${s%.*} B/s $u"; done'''
    r = ssh_cmd(receipt, script, timeout=120)
    out = r.stdout + r.stderr
    gpus = [l for l in out.splitlines() if l.startswith('GPU ')]
    ram = int(next((l.split('=')[1] for l in out.splitlines() if l.startswith('RAM_GB=')), '0') or 0)
    disk = int(next((l.split('=')[1] for l in out.splitlines() if l.startswith('DISK_GB=')), '0') or 0)
    pod_id = next((l.split('=')[1] for l in out.splitlines() if l.startswith('POD_ID=')), '')
    problems = []
    if pod_id != receipt['pod_id']:
        problems.append(f'pod id mismatch: {pod_id!r}')
    if len(gpus) != 4 or any('H200' not in g for g in gpus):
        problems.append(f'GPUs: {gpus}')
    if ram < 1000:
        problems.append(f'host RAM {ram} GB < 1000')
    if disk < 1900:
        problems.append(f'disk {disk} GB < 1900')
    return out, problems


def push_gcs_env(receipt):
    """/root/.gcs/gcs.env -> pod, over ssh stdin under umask 077; contents never printed or logged."""
    env_file = Path(C.GCS_ENV_FILE)
    if not env_file.is_file():
        print('WARNING: no /root/.gcs/gcs.env on the box; pod-side GCS pushes will be parked', flush=True)
        return False
    r = ssh_cmd(receipt, 'umask 077; mkdir -p /root/.gcs; cat > /root/.gcs/gcs.env; chmod 600 /root/.gcs/gcs.env; wc -c < /root/.gcs/gcs.env',
                timeout=60, stdin=env_file.read_text())
    if r.returncode or int((r.stdout.strip() or '0').split()[-1]) != env_file.stat().st_size:
        raise RuntimeError(f'GCS env push failed rc={r.returncode}')
    return True


def relaunch(worker):
    """Re-ship the current bundle to an already-provisioned pod and restart its worker (before any cell trained)."""
    receipt = json.loads(receipt_path(worker).read_text())
    log_dir = Path(receipt['log_dir'])
    launched_path = DEPLOY / f'{worker}.launched.json'
    root = f'{C.POD_ROOT}/{worker}'
    probe = ssh_cmd(receipt, f'ls {root}/cells 2>/dev/null | head -3; cat /workspace/glm-grid-phase.json 2>/dev/null', timeout=60)
    if probe.stdout.strip().split('\n')[0].startswith(C.PROFILE):
        raise RuntimeError(f'Cells already exist under {root}; refusing an automatic relaunch: {probe.stdout[:300]}')
    print('pod phase before relaunch:', probe.stdout.strip()[-200:], flush=True)
    deployed = bundle(force=True)
    write_json(log_dir / f'DEPLOYED_SOURCE.relaunch-{int(time.time())}.json', deployed)
    for name in ('base-code.tar.gz', 'code-overlay.tar.gz', 'prepared-inputs.tar.gz'):
        scp_to(receipt, DEPLOY / name, f'/workspace/{name}')
    token = subprocess.check_output([str(C.REPO / '.venv/bin/python'), '-c',
                                     'from huggingface_hub import get_token; print(get_token())'], text=True).strip()
    push_gcs_env(receipt)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    scp_to(receipt, C.HERE / 'ops/pod_relaunch.sh', '/workspace/glm_grid_relaunch.sh')
    remote = (f'bash /workspace/glm_grid_relaunch.sh {worker} {C.POD_PREPARED} {C.POD_ROOT} '
              f'/workspace/scimt/{C.HERE.relative_to(C.REPO)}/setup.sh')
    r = ssh_cmd(receipt, remote, timeout=900, stdin=token + '\n')
    (log_dir / f'relaunch-{stamp}.log').write_text(r.stdout + r.stderr)
    if r.returncode or 'LAUNCHED' not in r.stdout:
        raise RuntimeError(f'Relaunch failed rc={r.returncode}: {r.stdout[-1500:]} {r.stderr[-1500:]}')
    launched = json.loads(launched_path.read_text()) if launched_path.exists() else dict(receipt)
    launched.update(relaunched_at=now(), relaunch_output=r.stdout.strip().splitlines()[-6:], deployed=deployed['archives'],
                    root=root, worker_log='/workspace/glm-grid-worker.log', phase_file='/workspace/glm-grid-phase.json',
                    ssh=f"ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p {receipt['port']} root@{receipt['ip']}")
    write_json(launched_path, launched)
    write_json(log_dir / 'pod.json', launched)
    print(f"RELAUNCHED {worker} {receipt['pod_id']} {receipt['ip']}:{receipt['port']}\n" + '\n'.join(launched['relaunch_output']), flush=True)
    return 0


def provision(worker):
    receipt = json.loads(receipt_path(worker).read_text())
    log_dir = Path(receipt['log_dir'])
    if (DEPLOY / f'{worker}.launched.json').exists():
        raise RuntimeError('Already launched; inspect the pod instead of relaunching')
    if not receipt.get('ip'):
        print(f"waiting for an ssh endpoint on {receipt['pod_id']} (12 min cap) ...", flush=True)
        if wait_ssh(receipt) is None:
            print(f"ORPHAN: pod {receipt['pod_id']} has no ssh endpoint after 12 min. It IS registered with pod-own. "
                  f"Stop it now with a bare: runpodctl pod stop {receipt['pod_id']}", flush=True)
            return 3
    print(f"ssh endpoint {receipt['ip']}:{receipt['port']}; waiting for sshd ...", flush=True)
    for _ in range(60):
        try:
            if ssh_cmd(receipt, 'echo ready', timeout=40).stdout.strip() == 'ready':
                break
        except subprocess.TimeoutExpired:
            pass
        time.sleep(10)
    else:
        print(f"ORPHAN-LIKE: endpoint up but sshd not answering for 10 min on {receipt['pod_id']}; investigate or stop", flush=True)
        return 3
    out, problems = preflight(receipt)
    (log_dir / 'preflight.log').write_text(out)
    print(out, flush=True)
    if problems:
        print(f"PREFLIGHT FAILED: {problems}. Pod {receipt['pod_id']} left running for inspection; stop it if unusable.", flush=True)
        write_json(log_dir / 'preflight-failed.json', dict(problems=problems, at=now()))
        return 4
    deployed = bundle()
    write_json(log_dir / 'DEPLOYED_SOURCE.json', deployed)
    ssh_cmd(receipt, 'test ! -e /workspace/GLM_GRID_LAUNCHED.json && mkdir -p /workspace/scimt /workspace/glm-grid-prepared', check=True)
    for name in ('base-code.tar.gz', 'code-overlay.tar.gz', 'prepared-inputs.tar.gz'):
        print(f'shipping {name} ({(DEPLOY / name).stat().st_size / 1e6:.0f} MB)', flush=True)
        scp_to(receipt, DEPLOY / name, f'/workspace/{name}')
    scp_to(receipt, DEPLOY_GIT, '/workspace/glm_deployment_git.py')
    token = subprocess.check_output([str(C.REPO / '.venv/bin/python'), '-c',
                                     'from huggingface_hub import get_token; print(get_token())'], text=True).strip()
    if not token or len(token) < 20:
        raise RuntimeError('No Hub token on the box')
    push_gcs_env(receipt)
    remote = (f'set -e; set -a; source /etc/rp_environment; set +a; umask 077; cat > /root/.hf_token; umask 022; '
              f'export PATH=/root/.local/bin:$PATH; '
              f'for n in base-code code-overlay prepared-inputs; do echo "$n $(sha256sum /workspace/$n.tar.gz | cut -c1-16)"; done; '
              f'tar -xzf /workspace/base-code.tar.gz -C /workspace/scimt; tar -xzf /workspace/code-overlay.tar.gz -C /workspace/scimt; '
              f'python3 /workspace/glm_deployment_git.py; '
              f'tar -xzf /workspace/prepared-inputs.tar.gz -C /workspace/glm-grid-prepared; '
              f'test ! -e /workspace/GLM_GRID_LAUNCHED.json; cp /workspace/glm-grid-prepared/READY.json /workspace/GLM_GRID_LAUNCHED.json; '
              f'nohup setsid bash /workspace/scimt/{C.HERE.relative_to(C.REPO)}/setup.sh {worker} {C.POD_PREPARED} {C.POD_ROOT} '
              f'> /workspace/glm-grid-worker.log 2>&1 < /dev/null & echo LAUNCHED pid $!')
    r = ssh_cmd(receipt, remote, timeout=900, stdin=token + '\n')
    (log_dir / 'provision.log').write_text(r.stdout + r.stderr)
    if r.returncode or 'LAUNCHED' not in r.stdout:
        raise RuntimeError(f'Remote launch failed rc={r.returncode}: {r.stderr[-2000:]}')
    launched = dict(receipt, launched_at=now(), launched_epoch=time.time(), remote_output=r.stdout.strip().splitlines(),
                    deployed=deployed['archives'], worker_log='/workspace/glm-grid-worker.log',
                    root=f'{C.POD_ROOT}/{worker}', phase_file='/workspace/glm-grid-phase.json',
                    ssh=f"ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p {receipt['port']} root@{receipt['ip']}")
    write_json(DEPLOY / f'{worker}.launched.json', launched)
    write_json(log_dir / 'pod.json', launched)
    print(f"LAUNCHED {worker} {receipt['pod_id']} {receipt['ip']}:{receipt['port']} log_dir={log_dir}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['plan', 'create', 'provision', 'bundle', 'relaunch'])
    ap.add_argument('--worker', choices=sorted(C.WORKERS))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.action == 'plan':
        p = json.loads((PREPARED / 'plan.json').read_text())
        ready_inputs(PREPARED, p)
        print(plan_summary(p))
        me = myself()
        print('account:', json.dumps(spend_check(me)))
        print('create command: cd', C.REPO, '&& PYTHONPATH=.:src .venv/bin/python -m',
              'experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.launch create --worker <worker>')
        return 0
    if a.action == 'bundle':
        print(json.dumps(bundle(force=a.force), indent=1))
        return 0
    if not a.worker:
        ap.error('--worker required')
    if a.action == 'create':
        create(a.worker, a.dry_run)
        return 0
    if a.action == 'relaunch':
        return relaunch(a.worker)
    return provision(a.worker)


if __name__ == '__main__':
    raise SystemExit(main())
