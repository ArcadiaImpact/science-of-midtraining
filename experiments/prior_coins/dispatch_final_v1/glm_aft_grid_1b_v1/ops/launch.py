"""Box-side launcher for the GLM grid pods (one RunPod account; raw GraphQL for minMemoryInGb).

  python -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.ops.launch plan
  python -m ...ops.launch create    --worker glm-grid-1b-a [--dry-run] [--gpu "NVIDIA H200"]  # spend check, GraphQL create, pod-own add
  python -m ...ops.launch provision --worker glm-grid-1b-a                 # wait ssh (12 min cap), preflight, ship, start
  python -m ...ops.launch relaunch  --worker glm-grid-1b-a                 # re-ship the bundle to a provisioned pod (no cells yet)
  python -m ...ops.launch bundle                                           # (re)build the three tarballs only

Copied in spirit from ops/launch_glm_repair.py (#1c), with the pod spec taken from
config.POD (the primary, 4xB200 SECURE) or config.POD_FALLBACK when --gpu names
its gpuTypeId ("NVIDIA H200"): gpu type/count, template, container disk,
minMemoryInGb and allowedCudaVersions all come from the chosen spec; both public
keys go in PUBLIC_KEY, the Hub token is piped to /root/.hf_token under umask 077.
The receipt records the spec actually requested (pod=spec, gpu_requested, fallback).
Never stops or deletes a pod.  Creation refuses when the account would exceed
its hourly cap or already runs BUDGET['max_glm_pods'] GLM grid pods (two: never
a third pod this wave).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.ops.common import (
    DEPLOY, POD_OWN, PREPARED, PROGRESS_GLOB, clean_env, graphql, myself, now, pod_get, receipt_path,
    scp_to, spend_check, ssh_cmd, write_json)
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.prepare import ready_inputs

IMAGE_SUFFIXES = ('.png', '.pdf', '.svg', '.jpg', '.jpeg', '.gif')
DEPLOY_GIT = C.REPO / 'experiments/prior_coins/dispatch_final_v1/ops/glm_deployment_git.py'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pod_spec(gpu=None):
    """config.POD unless the fallback's gpuTypeId is named; any other string is an error."""
    if gpu is None or gpu == C.POD['gpu']:
        return C.POD
    if gpu == C.POD_FALLBACK['gpu']:
        return C.POD_FALLBACK
    raise ValueError(f"--gpu {gpu!r} is neither {C.POD['gpu']!r} nor the fallback {C.POD_FALLBACK['gpu']!r}")


def mutation(spec, name, public):
    """podFindAndDeployOnDemand built from a config pod spec (C.POD or C.POD_FALLBACK)."""
    return f'''mutation {{ podFindAndDeployOnDemand(input: {{
      cloudType: {spec["cloud"]}, gpuCount: {int(spec["gpu_count"])}, gpuTypeId: {json.dumps(spec["gpu"])},
      templateId: {json.dumps(spec["template"])}, containerDiskInGb: {int(spec["disk_gb"])},
      volumeInGb: 0, minMemoryInGb: {int(spec["min_host_ram_gb"])},
      allowedCudaVersions: {json.dumps([str(v) for v in spec["allowed_cuda_versions"]])},
      ports: "22/tcp,8888/http", startSsh: true, supportPublicIp: true,
      env: [{{key: "PUBLIC_KEY", value: {json.dumps(public)}}}], name: {json.dumps(name)}
    }}) {{ id machineId costPerHr }} }}'''


def field(lines, key, default=''):
    return next((line.split('=', 1)[1].strip() for line in lines if line.startswith(key + '=')), default)


def gpu_mib(line):
    """'GPU <name>, <memory.total MiB>, <memory.used MiB>' -> total MiB (0 when unparsable)."""
    parts = [x.strip() for x in line[len('GPU '):].split(',')]
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


def cuda_version(text):
    """'12.8' -> (12, 8); unparsable -> (0, 0), which fails any >= min_driver_cuda check."""
    try:
        parts = str(text).strip().split('.')
        return int(parts[0]), (int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        return (0, 0)


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
             f"pod: {C.POD['gpu_count']}x{C.POD['gpu']} {C.POD['cloud']} ${C.POD['hourly_usd']}/h; fallback after "
             f"{C.FALLBACK_AFTER_MIN} min without supply: {C.POD_FALLBACK['gpu_count']}x{C.POD_FALLBACK['gpu']} "
             f"${C.POD_FALLBACK['hourly_usd']}/h; budget: cap ${C.BUDGET['account_hourly_cap']}/h, "
             f"max {C.BUDGET['max_glm_pods']} GLM grid pods",
             f"hub: {p['publishing']['hub_repo']} / {p['publishing']['hub_prefix']}/<profile>/<arm>/<mix> (blobs only)",
             f"large files: mirror {p['publishing']['mirror_root']} ; gcs {p['publishing']['gcs_target']} (pending creds)"]
    for w, entry in p['workers'].items():
        pod = entry.get('pod') or C.POD
        lines.append(f"{w} (slot {C.WORKER_SLOT.get(w, '?')}, {entry['arm']}, pod {entry['pod_name']}: "
                     f"{pod['gpu_count']}x{pod['gpu']} ${pod['hourly_usd']}/h): " + ' -> '.join(
            f"{j['mix']}[{j['conflict_rows']}]" for j in entry['jobs']))
    lines.append('datasets: ' + ', '.join(f"{m} {d['sha256'][:10]}" for m, d in p['datasets'].items()))
    lines.append('tokens/row (GLM, measured): ' + ', '.join(f"{m} {t['tokens_per_row']}" for m, t in p['tokens'].items()))
    lines.append(f"pinned sources: {len(p['source_hashes'])} files")
    return '\n'.join(lines)


def create(worker, dry_run, gpu=None):
    spec = pod_spec(gpu)
    fallback = spec is C.POD_FALLBACK
    p = json.loads((PREPARED / 'plan.json').read_text())
    ready_inputs(PREPARED, p)
    entry = p['workers'][worker]
    name = entry['pod_name']
    me = myself()
    report = spend_check(me, spec['hourly_usd'])
    if any(pod['name'] == name for pod in me['pods']):
        raise RuntimeError(f'A pod named {name} already exists (any state); inspect, never duplicate')
    if receipt_path(worker).exists() or (DEPLOY / f'{worker}.pending.json').exists():
        raise RuntimeError('Existing/ambiguous allocation receipt; inspect, never duplicate')
    public = (Path.home() / '.ssh/id_ed25519.pub').read_text().strip() + '\n' + \
        (Path.home() / '.runpod/ssh/runpodctl-ssh-key.pub').read_text().strip()
    body = mutation(spec, name, public)
    log_dir = C.LOG_ROOT / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{C.WORKER_SLOT[worker]}"
    print(json.dumps(dict(worker=worker, pod_name=name, gpu=spec['gpu'], hourly_usd=spec['hourly_usd'], fallback=fallback,
                          spend=report), indent=1))
    if dry_run:
        print('DRY RUN: would POST podFindAndDeployOnDemand:\n' + body.replace(json.dumps(public), '"<public keys>"'))
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(DEPLOY / f'{worker}.pending.json', dict(worker=worker, name=name, gpu=spec['gpu'], spend=report, at=now(),
                                                      plan_sha256=p and hashlib.sha256((PREPARED / 'plan.json').read_bytes()).hexdigest()))
    try:
        result = graphql(body).get('podFindAndDeployOnDemand')
    except RuntimeError as error:
        if 'SUPPLY_CONSTRAINT' in str(error):
            # Definitive "nothing was created": clear the intent so a snipe loop can retry.
            (DEPLOY / f'{worker}.pending.json').unlink(missing_ok=True)
            print(f'SUPPLY_CONSTRAINT {worker} {spec["gpu"]} {now()}', flush=True)
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
    receipt = dict(worker=worker, arm=entry['arm'], name=name, pod_id=result['id'], hourly=result['costPerHr'],
                   machine_id=result.get('machineId'), created_at=now(), created_epoch=time.time(),
                   log_dir=str(log_dir), spend_before=report, pod=spec, gpu_requested=spec['gpu'], fallback=fallback)
    write_json(receipt_path(worker), receipt)
    (DEPLOY / f'{worker}.pending.json').unlink()
    write_json(log_dir / 'pod-create.json', dict(receipt, graphql_result=result))
    own = subprocess.run([str(POD_OWN), 'add', result['id'], '--progress-glob', PROGRESS_GLOB, '--stale-min', '90'],
                         env=clean_env(), capture_output=True, text=True)
    (log_dir / 'pod-own.log').write_text(own.stdout + own.stderr)
    print(f"POD CREATED {result['id']} name={name} gpu={spec['gpu']} ${result['costPerHr']}/h machine={result.get('machineId')} "
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


PREFLIGHT_SCRIPT = r'''set -a; source /etc/rp_environment 2>/dev/null; set +a
echo POD_ID=$RUNPOD_POD_ID
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | sed 's/^/GPU /'
echo DRIVER_CUDA=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n 1)
echo COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | paste -sd, -)
echo RAM_GB=$(free -g | awk '/Mem:/{print $2}')
echo CGROUP_MAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo none)
echo DISK_GB=$(df -BG /workspace | awk 'NR==2{print $4}' | tr -d G)
echo PY=$(python3 --version 2>&1)
echo TORCH=$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)
echo GIT=$(command -v git || echo none) RSYNC=$(command -v rsync || echo none)
PYPI=$(curl -sS --max-time 20 https://pypi.org/pypi/nvidia-cudnn-cu12/json | python3 -c 'import json,sys; u=[i for i in json.load(sys.stdin).get("urls",[]) if i["filename"].endswith(".whl")]; print(max(u,key=lambda i:i.get("size",0))["url"])' 2>/dev/null || echo https://files.pythonhosted.org/)
for u in 'https://download.pytorch.org/whl/cu126/torch-2.12.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl' "$PYPI" 'https://huggingface.co/zai-org/GLM-4.5-Air-Base/resolve/main/model-00001-of-00046.safetensors'; do
  s=$(curl -sS -o /dev/null -w '%{speed_download}' --max-time 15 -r '0-300000000' "$u" 2>/dev/null || echo 0); echo "PROBE ${s%.*} B/s $u"; done'''


def preflight_problems(receipt, out):
    """Check the preflight output against the spec recorded in the receipt (receipt['pod'] = C.POD or C.POD_FALLBACK)."""
    pod = receipt['pod']
    lines = out.splitlines()
    gpus = [line for line in lines if line.startswith('GPU ')]
    ram = int(field(lines, 'RAM_GB', '0') or 0)
    disk = int(field(lines, 'DISK_GB', '0') or 0)
    driver = field(lines, 'DRIVER_CUDA')
    problems = []
    if field(lines, 'POD_ID') != receipt['pod_id']:
        problems.append(f"pod id mismatch: {field(lines, 'POD_ID')!r}")
    if len(gpus) != pod['gpu_count'] or any(pod['gpu_family'] not in g for g in gpus):
        problems.append(f"GPUs: {gpus} (want {pod['gpu_count']} x {pod['gpu_family']})")
    small = [g for g in gpus if gpu_mib(g) < C.GPU_MIN_MEMORY_GIB * 1024]
    if small:
        problems.append(f'GPU memory below {C.GPU_MIN_MEMORY_GIB} GiB: {small}')
    if ram < pod['min_host_ram_gb']:
        problems.append(f"host RAM {ram} GB < {pod['min_host_ram_gb']}")
    if disk < 1900:
        problems.append(f'disk {disk} GB < 1900')
    if cuda_version(driver) < cuda_version(pod['min_driver_cuda']):
        problems.append(f"driver CUDA {driver or '?'} < {pod['min_driver_cuda']} (compute_cap {field(lines, 'COMPUTE_CAP') or '?'})")
    return problems


def preflight(receipt):
    r = ssh_cmd(receipt, PREFLIGHT_SCRIPT, timeout=120)
    out = r.stdout + r.stderr
    return out, preflight_problems(receipt, out)


def push_gcs_env(receipt):
    """/root/.gcs/gcs.env -> pod, over ssh stdin under umask 077; contents never printed or logged."""
    env_file = Path(C.GCS_ENV_FILE)
    if not env_file.is_file():
        print(f'WARNING: no {C.GCS_ENV_FILE} on the box; pod-side GCS pushes will be parked', flush=True)
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
              f'{C.POD_REPO}/{C.HERE.relative_to(C.REPO)}/setup.sh {C.POD_REPO}')
    r = ssh_cmd(receipt, remote, timeout=900, stdin=token + '\n')
    (log_dir / f'relaunch-{stamp}.log').write_text(r.stdout + r.stderr)
    if r.returncode or 'LAUNCHED' not in r.stdout:
        raise RuntimeError(f'Relaunch failed rc={r.returncode}: {r.stdout[-1500:]} {r.stderr[-1500:]}')
    launched = json.loads(launched_path.read_text()) if launched_path.exists() else dict(receipt)
    launched.update(relaunched_at=now(), relaunch_output=r.stdout.strip().splitlines()[-6:], deployed=deployed['archives'],
                    gpu=receipt['pod']['gpu'], root=root, worker_log='/workspace/glm-grid-worker.log', phase_file='/workspace/glm-grid-phase.json',
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
    ssh_cmd(receipt, f'test ! -e /workspace/GLM_GRID_LAUNCHED.json && mkdir -p {C.POD_REPO} {C.POD_PREPARED}', check=True)
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
              f'tar -xzf /workspace/base-code.tar.gz -C {C.POD_REPO}; tar -xzf /workspace/code-overlay.tar.gz -C {C.POD_REPO}; '
              f'python3 /workspace/glm_deployment_git.py; '
              f'tar -xzf /workspace/prepared-inputs.tar.gz -C {C.POD_PREPARED}; '
              f'test ! -e /workspace/GLM_GRID_LAUNCHED.json; cp {C.POD_PREPARED}/READY.json /workspace/GLM_GRID_LAUNCHED.json; '
              f'nohup setsid bash {C.POD_REPO}/{C.HERE.relative_to(C.REPO)}/setup.sh {worker} {C.POD_PREPARED} {C.POD_ROOT} '
              f'> /workspace/glm-grid-worker.log 2>&1 < /dev/null & echo LAUNCHED pid $!')
    r = ssh_cmd(receipt, remote, timeout=900, stdin=token + '\n')
    (log_dir / 'provision.log').write_text(r.stdout + r.stderr)
    if r.returncode or 'LAUNCHED' not in r.stdout:
        raise RuntimeError(f'Remote launch failed rc={r.returncode}: {r.stderr[-2000:]}')
    launched = dict(receipt, launched_at=now(), launched_epoch=time.time(), gpu=receipt['pod']['gpu'],
                    remote_output=r.stdout.strip().splitlines(),
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
    ap.add_argument('--gpu', default=None,
                    help=f"gpuTypeId to request: {C.POD['gpu']} (default) or the fallback {C.POD_FALLBACK['gpu']}")
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
              'experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.ops.launch create --worker <worker>',
              f'[--gpu "{C.POD_FALLBACK["gpu"]}"]')
        return 0
    if a.action == 'bundle':
        print(json.dumps(bundle(force=a.force), indent=1))
        return 0
    if not a.worker:
        ap.error('--worker required')
    if a.action == 'create':
        create(a.worker, a.dry_run, gpu=a.gpu)
        return 0
    if a.action == 'relaunch':
        return relaunch(a.worker)
    return provision(a.worker)


if __name__ == '__main__':
    raise SystemExit(main())
