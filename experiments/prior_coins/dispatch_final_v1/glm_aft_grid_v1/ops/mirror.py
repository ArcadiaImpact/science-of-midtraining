"""Pull finished cells from a pod to the driver box and verify them (the large-file fail-safe).

  python -m ...ops.mirror --worker glm-grid-charter            # every cell with MIRROR_READY.json not yet verified locally
  python -m ...ops.mirror --worker glm-grid-charter --fsdp     # also the FSDP recovery states (checkpoints/, ~12 GB/cell)
  python -m ...ops.mirror --worker glm-grid-charter --worker-files   # worker-root receipts + setup/worker logs -> log dir

Destination: <MIRROR_ROOT>/<arm>/<mix>/ (adapters/, eval/, receipts, logs,
aft_<mix>.jsonl).  Adapter safetensors are sha256-checked against each
EXPORT_COMPLETE.json, the dataset against DATASET.json; MIRROR_VERIFIED.json
is written only when everything matches.  Read-only on the pod.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.common import (
    DEPLOY, SSH_OPTS, ssh_cmd, write_json)
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import sha


def rsync(receipt, remote, local, excludes=()):
    local.mkdir(parents=True, exist_ok=True)
    cmd = ['env', '-u', 'SSH_AUTH_SOCK', 'rsync', '-a', '--partial', '--inplace', '--timeout=600',
           '-e', 'ssh ' + ' '.join(SSH_OPTS) + f" -p {receipt['port']}"]
    for e in excludes:
        cmd += ['--exclude', e]
    cmd += [f"root@{receipt['ip']}:{remote}/", str(local) + '/']
    for attempt in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode == 0:
            return
        time.sleep(30)
    raise RuntimeError(f'rsync failed rc={r.returncode}: {r.stderr[-1500:]}')


def verify_cell(local, mix):
    problems = []
    for step in C.SAVES:
        cp = local / 'adapters' / f'step{step}'
        receipt = cp / 'EXPORT_COMPLETE.json'
        weights = cp / 'adapter_model.safetensors'
        if not receipt.exists() or not weights.exists():
            problems.append(f'step{step}: missing export')
            continue
        if sha(weights) != json.loads(receipt.read_text())['sha256']:
            problems.append(f'step{step}: adapter sha mismatch')
    dataset = local / f'aft_{mix}.jsonl'
    if not dataset.exists() or sha(dataset) != json.loads((local / 'DATASET.json').read_text())['sha256']:
        problems.append('dataset sha mismatch/missing')
    for step in C.EVAL_STEPS:
        if not (local / 'eval' / f'{mix}-step{step}' / 'scores.json').exists():
            problems.append(f'eval step{step}: scores.json missing')
    if not (local / 'COMPLETE.json').exists():
        problems.append('COMPLETE.json missing')
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', required=True, choices=sorted(C.WORKERS))
    ap.add_argument('--fsdp', action='store_true')
    ap.add_argument('--worker-files', action='store_true')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    receipt = json.loads((DEPLOY / f'{a.worker}.launched.json').read_text())
    arm = C.WORKERS[a.worker]
    root = f"{C.POD_ROOT}/{a.worker}"
    log_dir = Path(receipt['log_dir'])
    if a.worker_files:
        dest = log_dir / 'pod-files'
        dest.mkdir(exist_ok=True)
        rsync(receipt, root, dest / 'worker-root', excludes=['cells/', 'parent/', 'eval-runtime/', '*.lock'])
        for f in ('/workspace/glm-grid-worker.log', '/workspace/setup.log', '/workspace/glm-grid-phase.json',
                  '/workspace/DEPLOYED_SOURCE_GIT.json', '/workspace/GLM_GRID_LAUNCHED.json'):
            subprocess.run(['env', '-u', 'SSH_AUTH_SOCK', 'scp', *SSH_OPTS, '-P', str(receipt['port']),
                            f"root@{receipt['ip']}:{f}", str(dest)], capture_output=True, timeout=600)
        print(f'worker files -> {dest}')
    out = ssh_cmd(receipt, f'ls {root}/cells/{C.PROFILE}/{arm}/*/MIRROR_READY.json 2>/dev/null', timeout=60).stdout
    ready = [Path(p).parent.name for p in out.split()]
    done = []
    for mix in ready:
        local = C.MIRROR_ROOT / arm / mix
        if (local / 'MIRROR_VERIFIED.json').exists() and not a.force and not a.fsdp:
            continue
        remote = f'{root}/cells/{C.PROFILE}/{arm}/{mix}'
        excludes = ['eval-work-*/', '*.lock'] + ([] if a.fsdp else ['checkpoints/'])
        started = time.time()
        rsync(receipt, remote, local, excludes)
        problems = verify_cell(local, mix)
        size = sum(p.stat().st_size for p in local.rglob('*') if p.is_file())
        if problems:
            write_json(local / 'MIRROR_PROBLEMS.json', dict(problems=problems, at=time.time()))
            print(f'MIRROR PROBLEMS {arm}/{mix}: {problems}', flush=True)
            continue
        write_json(local / 'MIRROR_VERIFIED.json', dict(job=f'{C.PROFILE}/{arm}/{mix}', pod_id=receipt['pod_id'],
                                                        bytes=size, fsdp_included=a.fsdp, seconds=round(time.time() - started),
                                                        adapters={str(s): json.loads((local / 'adapters' / f'step{s}' / 'EXPORT_COMPLETE.json').read_text())['sha256'] for s in C.SAVES},
                                                        at=time.time()))
        done.append(mix)
        print(f'MIRRORED {arm}/{mix} {size / 1e9:.2f} GB in {time.time() - started:.0f}s -> {local}', flush=True)
    print(json.dumps(dict(worker=a.worker, ready=ready, mirrored_now=done,
                          verified=sorted(p.parent.name for p in (C.MIRROR_ROOT / arm).glob('*/MIRROR_VERIFIED.json')))))


if __name__ == '__main__':
    main()
