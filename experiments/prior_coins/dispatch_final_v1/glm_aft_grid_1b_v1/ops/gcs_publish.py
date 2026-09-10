"""Publish mirrored GLM grid cells (the large files) to GCS with rclone; dry run by default.

  python -m ...ops.gcs_publish --dry-run                                   # plan only (no credentials needed)
  python -m ...ops.gcs_publish --execute                                   # copy + check, GCS_PUBLISHED.json per cell
  python -m ...ops.gcs_publish --target gs://other-bucket/prefix ...       # override the default target

Source: <MIRROR_ROOT>/<arm>/<mix>/ cells carrying MIRROR_VERIFIED.json.  The
whole verified cell directory is copied (adapters, responses, receipts,
training JSONL; FSDP states only if mirrored) to <target>/<arm>/<mix>/, then
`rclone check` confirms every object (md5).  A GCS_PUBLISHED.json manifest
(sha256 per adapter, gs:// URIs) is written beside the cell; --hub also pushes
that small manifest to the cell's Hub prefix so the Hub points at GCS.
Credentials: RCLONE_CONFIG_GCS_* sourced by bash from --env-file (default /root/.gcs/gcs.env,
never on argv or in logs); the interactive shell's stale RCLONE_CONFIG_GCS_* copies are dropped first.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.ops.common import write_json


#: Box-side excludes: the pinned set plus files that change while rclone runs or are box/pod bookkeeping.
BOX_EXCLUDES = (*C.GCS_EXCLUDES, 'gcs.log', 'GCS_*.json', 'MIRROR_*.json')


def cells(source):
    for verified in sorted(source.glob('*/*/MIRROR_VERIFIED.json')):
        cell = verified.parent
        yield cell.parent.name, cell.name, cell


def inventory(cell):
    files = {}
    for p in sorted(cell.rglob('*')):
        if p.is_file() and not p.name.startswith(('GCS_', 'MIRROR_')) and p.name != 'gcs.log' and 'receipts' not in p.parts:
            files[str(p.relative_to(cell))] = p.stat().st_size
    return files


def rclone(env_file, args, **kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith('RCLONE_CONFIG_GCS_')}
    argv = ['bash', '-c', 'set -a; . "$0"; set +a; exec rclone --config /dev/null "$@"', str(env_file), *map(str, args)]
    return subprocess.run(argv, env=env, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, default=C.MIRROR_ROOT)
    ap.add_argument('--target', default=C.GCS_TARGET, help='gs://bucket/prefix')
    ap.add_argument('--env-file', default=C.GCS_ENV_FILE)
    ap.add_argument('--execute', action='store_true', help='actually copy (default: dry run)')
    ap.add_argument('--dry-run', action='store_true', help='explicit dry run (the default)')
    ap.add_argument('--hub', action='store_true', help='after a verified copy, publish GCS_PUBLISHED.json to the Hub cell prefix')
    ap.add_argument('--only', help='<arm>/<mix> to publish just one cell')
    a = ap.parse_args()
    if not a.target.startswith('gs://'):
        raise SystemExit('target must be a gs:// URI')
    bucket_path = a.target[len('gs://'):].rstrip('/')
    execute = a.execute and not a.dry_run
    if execute and not Path(a.env_file).is_file():
        raise SystemExit(f'No GCS credentials file: {a.env_file}')
    total = 0
    plan = {}
    for arm, mix, cell in cells(a.source):
        if a.only and a.only != f'{arm}/{mix}':
            continue
        files = inventory(cell)
        size = sum(files.values())
        total += size
        uri = f'{a.target}/{C.PROFILE}/{arm}/{mix}'
        adapters = {str(s): json.loads((cell / 'adapters' / f'step{s}' / 'EXPORT_COMPLETE.json').read_text())['sha256']
                    for s in C.SAVES if (cell / 'adapters' / f'step{s}' / 'EXPORT_COMPLETE.json').exists()}
        plan[f'{arm}/{mix}'] = dict(files=len(files), bytes=size, target=uri, adapters=adapters)
        remote = f'gcs:{bucket_path}/{C.PROFILE}/{arm}/{mix}'
        cmd = ['copy', '--transfers', '8', '--checkers', '16', '--stats-one-line', '--stats', '30s', *[x for e in BOX_EXCLUDES for x in ('--exclude', e)],
               str(cell), remote]
        if not execute:
            print(f'DRY RUN {arm}/{mix}: {len(files)} files, {size / 1e9:.2f} GB -> {uri}')
            print('    rclone --config /dev/null', ' '.join(cmd))
            write_json(cell / 'GCS_PLAN.json', dict(target=uri, files=files, bytes=size, planned=time.time(), executed=False))
            continue
        if (cell / 'GCS_PUBLISHED.json').exists():
            print(f'already published {arm}/{mix}')
            continue
        started = time.time()
        rclone(a.env_file, cmd, check=True)
        check = rclone(a.env_file, ['check', '--one-way', *[x for e in BOX_EXCLUDES for x in ('--exclude', e)], str(cell), remote],
                       capture_output=True, text=True)
        if check.returncode:
            raise RuntimeError(f'rclone check failed for {arm}/{mix}: {check.stderr[-2000:]}')
        published = dict(target=uri, bucket_path=f'{bucket_path}/{C.PROFILE}/{arm}/{mix}', files={k: dict(size=v, uri=f'{uri}/{k}') for k, v in files.items()},
                         bytes=size, adapters_sha256=adapters, seconds=round(time.time() - started), at=time.time(),
                         verified='rclone check --one-way (md5) from the sha-verified box mirror', excludes=list(BOX_EXCLUDES),
                         source='box mirror (the pod-side push lands first; this box-side rclone check is the authoritative verification)')
        write_json(cell / 'GCS_PUBLISHED.json', published)
        print(f'PUBLISHED {arm}/{mix} {size / 1e9:.2f} GB -> {uri} in {time.time() - started:.0f}s', flush=True)
        if a.hub:
            from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import Publisher
            pub = Publisher(C.MODEL_REPO, f'{C.HUB_PREFIX}/{C.PROFILE}/{arm}/{mix}', cell / 'receipts')
            pub.publish(cell, [cell / 'GCS_PUBLISHED.json'], 'gcs-published')
            print(f'   Hub manifest published under {C.HUB_PREFIX}/{C.PROFILE}/{arm}/{mix}/GCS_PUBLISHED.json')
    print(json.dumps(dict(cells=len(plan), bytes=total, gb=round(total / 1e9, 2), target=a.target,
                          mode='execute' if execute else 'dry-run'), indent=1))


if __name__ == '__main__':
    main()
