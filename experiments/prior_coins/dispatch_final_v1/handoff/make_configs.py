"""Emit a pod config for handoff/launch_halfpct_pod.sh from the cell inventory.

    python3 make_configs.py --name jb-halfpct-27b-a --gpu "NVIDIA H200" --disk 500 \
        --workers A3-27b-half02 --out configs/27b-a.json

One pod runs the listed workers in order.  A worker's jobs are its inventory
cells with action jonathan_train_eval / jonathan_eval_only; cells still on
Sid's queues are excluded unless --ownership-confirmed.  A worker with an
evaluation-only cell is restored from its recovery archive at the original
root; a worker whose remaining cell had an interrupted attempt runs fresh in a
new root and publishes under the --attempt-suffix namespace.
"""
import argparse
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
INVENTORY = HERE.parent / 'JONATHAN_GEMMA_HALFPCT_CELLS.json'
# The campaign created its pods from RunPod template runpod-torch-v280 (see
# ops/launch_gemma_halfpct.py); this is that template's image.  It matters:
# pod/setup.sh installs into the system Python and pins a CPython 3.12
# flash-attn wheel, and scimt itself needs Python >= 3.11 -- the 3.10-based
# torch271 image fails setup at `uv pip install -e .`.
IMAGE = 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
MIN_CUDA_VERSION = '12.8'  # cu1281 image; the campaign asked for 12.8-13.1
# Sid released his queue (Slack, relayed by Jonathan 2026-09-08 19:40Z). These five cells
# live in his A2 workers of plan gemma-aft-halfpct-balanced-v1 (mapping mirrors plan.json;
# the wrapper re-validates job-in-worker at runtime). They have no recovery archive, so the
# code bundles and the worker-independent shared-data receipt come from a donor archive.
RELEASED_WORKERS = {
    'A2-27b-half01': 'gemma3_27b_5m/charter/charter_0p5pct',
    'A2-27b-half03': 'gemma3_27b_19m/charter/charter_0p5pct',
    'A2-27b-half05': 'gemma3_27b_50m/charter/charter_0p5pct',
    'A2-27b-half07': 'gemma3_27b_190m/charter/charter_0p5pct',
    'A2-27b-half09': 'gemma3_27b_5m/control/charter_0p5pct'}
SID_QUEUED = {
    'gemma3_27b_5m/charter/charter_0p5pct', 'gemma3_27b_19m/charter/charter_0p5pct',
    'gemma3_27b_50m/charter/charter_0p5pct', 'gemma3_27b_190m/charter/charter_0p5pct',
    'gemma3_27b_5m/control/charter_0p5pct'}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--name', required=True)
    p.add_argument('--gpu', required=True, help='runpod gpu id, e.g. "NVIDIA H200"')
    p.add_argument('--disk', type=int, required=True, help='container disk GB')
    p.add_argument('--cloud', default='SECURE')
    p.add_argument('--workers', nargs='+', required=True)
    p.add_argument('--attempt-suffix', default='-jonathan-rerun1')
    p.add_argument('--ownership-confirmed', action='store_true')
    p.add_argument('--parent-revision-override', default=None, metavar='SHA',
                   help='full commit sha of arcadia-impact/scimt-dispatch-final-v1 to re-fetch archived '
                        "cells' parents from (the plan's pinned revision was squashed away 2026-09-09); "
                        'exported by the bootstrap as PARENT_REVISION_OVERRIDE, accepted by the wrapper '
                        'only with byte-identical weights/tokenizer/config (see RESTORE_PARENT.json)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.parent_revision_override is not None and not re.fullmatch(r'[0-9a-f]{40}', a.parent_revision_override):
        raise SystemExit('--parent-revision-override must be a full 40-hex commit sha')
    inventory = json.loads(INVENTORY.read_text())
    by_worker = {}
    for cell in inventory['cells']:
        archive = cell.get('recovery_archive')
        if archive:
            by_worker.setdefault(archive['prefix'].rsplit('/', 1)[1], []).append(cell)
    runs = []
    models = set()
    by_id = {c['id']: c for c in inventory['cells']}
    for worker in a.workers:
        if worker in RELEASED_WORKERS:
            if not a.ownership_confirmed:
                raise SystemExit(f'{worker} holds a released Sid cell; pass --ownership-confirmed')
            cell = by_id[RELEASED_WORKERS[worker]]
            model = worker.split('-')[1]
            models.add(model)
            donor = next(c for c in inventory['cells'] if c.get('recovery_archive')
                         and c['id'].startswith(f'gemma3_{model}'))
            archive = donor['recovery_archive']
            runs.append(dict(
                worker=worker, mode='fresh', profile=cell['id'].split('/')[0],
                archive=dict(repo=archive['repo'], prefix=archive['prefix'], commit=archive['commit'],
                             manifest_sha256=archive['files']['MANIFEST.json']['sha256'],
                             tar_sha256=archive['files']['partial-work.tar']['sha256']),
                receipt_worker=archive['prefix'].rsplit('/', 1)[1],
                # fresh root per attempt: a re-attempt after an interrupted run must not reuse the root
                root=f"/workspace/gemma-halfpct{a.attempt_suffix.replace('-jonathan', '')}/{worker}", jobs=[cell['id']],
                resume_eval=False, attempt_suffix=a.attempt_suffix, ownership_confirmed=True,
                inventory_actions={cell['id']: cell['action']},
                released='Sid via Slack, relayed by Jonathan 2026-09-08'))
            continue
        cells = by_worker[worker]
        model = worker.split('-')[1]
        models.add(model)
        mine = [c for c in cells if c['action'] in ('jonathan_train_eval', 'jonathan_eval_only')
                and (a.ownership_confirmed or c['id'] not in SID_QUEUED)]
        if not mine:
            raise SystemExit(f'{worker}: nothing to run')
        for c in mine:
            if c['ownership_confirm_before_start'] and not a.ownership_confirmed:
                raise SystemExit(f"{c['id']} needs ownership confirmation")
        eval_only = [c for c in mine if c['action'] == 'jonathan_eval_only']
        interrupted = [c for c in mine if c['action'] == 'jonathan_train_eval'
                       and c.get('recovery_archive', {}).get('interrupted_status', {}).get('job') == c['id']
                       and c.get('recovery_archive', {}).get('interrupted_status', {}).get('stage') == 'train']
        mode = 'restore' if eval_only else 'fresh'
        archive = cells[0]['recovery_archive']
        profile = mine[0]['id'].split('/')[0]
        runs.append(dict(
            worker=worker, mode=mode, profile=profile,
            archive=dict(repo=archive['repo'], prefix=archive['prefix'], commit=archive['commit'],
                         manifest_sha256=archive['files']['MANIFEST.json']['sha256'],
                         tar_sha256=archive['files']['partial-work.tar']['sha256']),
            root=(f'/workspace/gemma-halfpct/{worker}' if mode == 'restore'
                  else f'/workspace/gemma-halfpct-rerun1/{worker}'),
            jobs=[c['id'] for c in mine],
            resume_eval=bool(eval_only),
            attempt_suffix=(a.attempt_suffix if interrupted else None),
            ownership_confirmed=a.ownership_confirmed,
            inventory_actions={c['id']: c['action'] for c in mine},
        ))
        if interrupted and mode == 'restore':
            raise SystemExit(f'{worker}: cannot mix a restore with an interrupted rerun')
    if len(models) != 1:
        raise SystemExit('one pod runs one model size')
    model = models.pop()
    config = dict(
        pod=dict(name=a.name, gpu_id=a.gpu, disk_gb=a.disk, cloud_type=a.cloud, image=IMAGE,
                 min_cuda_version=MIN_CUDA_VERSION),
        model=model, setup_profile=runs[0]['profile'],
        publish_repo=f'arcadia-impact/scimt-dispatch-gemma-{model}-aft-grid-v2',
        inventory_checked_at=inventory['checked_at'], runs=runs,
    )
    if a.parent_revision_override:
        config['parent_revision_override'] = a.parent_revision_override
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
