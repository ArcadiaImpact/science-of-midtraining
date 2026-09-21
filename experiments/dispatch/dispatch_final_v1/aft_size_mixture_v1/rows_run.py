"""Versioned row-dose runner, preserving original agreement training and provenance."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import run
from rows_v2 import CELLS, SCHEDULES, validate
from shards import completed_training, proc


def bind(path, payload):
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError(f'Immutable identity mismatch: {path}')
    run.write(path, payload)


def same(child):
    p = proc(child['pid'])
    return p is not None and p['start'] == child['start']


def publish(dest, arm, cell):
    from huggingface_hub import HfApi
    result = HfApi().upload_folder(
        repo_id=run.MODEL_REPO, folder_path=dest,
        path_in_repo=f'followups/aft-size-mixture-rows-v2/{arm}/{cell}',
        allow_patterns=['adapters/**', 'eval/**', '*.json', '*.log', '*.yaml', 'health/**'],
        commit_message=f'AFT row doses v2 {arm}/{cell}: both epoch evaluations')
    run.write(dest / 'PUBLISHED.json', {'repo': run.MODEL_REPO, 'commit': result.oid})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=run.ARMS, required=True)
    ap.add_argument('--shard', choices=SCHEDULES, required=True)
    ap.add_argument('--data', type=Path, default=Path('/workspace/aft-size-data-rows-v2'))
    ap.add_argument('--root', type=Path, default=Path('/workspace/aft-size-mixture-rows-v2'))
    ap.add_argument('--legacy-root', type=Path, default=Path('/workspace/aft-size-mixture-v1'))
    ap.add_argument('--disable-nvls', action='store_true')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--train-cell', choices=[c[0] for c in CELLS])
    ap.add_argument('--parent', type=Path)
    a = ap.parse_args()
    validate(a.data)
    if a.train_cell:
        assert a.parent is not None
        run.train(a.train_cell, a.parent, a.data, a.root)
        return
    cells = SCHEDULES[a.shard]
    print(json.dumps({'arm': a.arm, 'shard': a.shard, 'cells': cells, 'dose_unit': 'rows'}), flush=True)
    if not a.execute:
        return
    if a.disable_nvls:
        os.environ['NCCL_NVLS_ENABLE'] = '0'
    root, old = a.root.resolve() / a.arm, a.legacy_root.resolve() / a.arm
    root.mkdir(parents=True, exist_ok=True)
    old.mkdir(parents=True, exist_ok=True)
    lock = (root / 'runner.lock').open('a')
    legacy_lock = (old / 'runner.lock').open('a')
    for f in (lock, legacy_lock):
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {**run.identity(a.arm, a.data), 'study': 'aft_size_mixture_rows_v2',
                'row_runner_sources': {p: run.sha(Path(__file__).with_name(p)) for p in ('rows_run.py', 'rows_v2.py')},
                'nccl_nvls_enable': os.environ.get('NCCL_NVLS_ENABLE', 'auto')}
    bind(root / 'IDENTITY.json', identity)
    bind(root / 'SHARD_PLAN.json', {'schema': 'aft_rows_v2', 'account': a.shard,
                                  'arm': a.arm, 'cells': list(cells), 'identity': identity})
    if a.shard == 'A1':
        stop = json.loads((old / 'TOKEN_MIGRATION_STOP.json').read_text())
        assert stop['status'] == 'stopped' and stop['shard'] == 'A1' and stop['arm'] == a.arm
        child = stop['agreement_child_preserved']
        agreement = old / 'agreement'
        if not (root / 'agreement').exists():
            (root / 'agreement').symlink_to(agreement, target_is_directory=True)
        assert (root / 'agreement').resolve() == agreement
        print('WAIT: original agreement training continues unchanged', flush=True)
        while same(child):
            time.sleep(30)
        completed_training(agreement)
        run.verify_adapters(agreement)
        if not (agreement / 'TRAIN_COMPLETE.json').exists():
            run.write(agreement / 'TRAIN_COMPLETE.json', {'steps': 5120,
                      'identity': json.loads((old / 'IDENTITY.json').read_text()),
                      'adopted_training': child})
    run.hardware_check(root)
    parent = run.fetch_parent(a.arm, old)
    for cell in cells:
        dest = root / cell
        dest.mkdir(exist_ok=True)
        if cell != 'agreement':
            bind(dest / 'IDENTITY.json', identity)
        bind(dest / 'ROWS_V2_EXECUTION.json', {'shard': a.shard, 'arm': a.arm, 'identity': identity})
        if not (dest / 'TRAIN_COMPLETE.json').exists():
            if (dest / 'training_started.json').exists():
                raise RuntimeError('Interrupted training requires a verified resume; refusing fresh restart')
            print(f'TRAIN {a.arm}/{cell}', flush=True)
            run.command([sys.executable, Path(__file__), '--arm', a.arm, '--shard', a.shard,
                         '--train-cell', cell, '--parent', parent, '--data', a.data, '--root', dest],
                        dest / 'driver.log', {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
            completed_training(dest)
            run.verify_adapters(dest)
            run.write(dest / 'TRAIN_COMPLETE.json', {'steps': 5120, 'identity': identity})
        else:
            completed_training(dest)
            run.verify_adapters(dest)
        if not (dest / 'EVAL_COMPLETE.json').exists():
            print(f'EVAL {a.arm}/{cell}', flush=True)
            run.evaluate(cell, parent, a.data, root, dest, '/workspace/venv-dispatch-eval/bin/python')
            run.write(dest / 'EVAL_COMPLETE.json', {'steps': list(run.EVAL_STEPS), 'identity': identity})
        if not (dest / 'PUBLISHED.json').exists():
            if cell == 'agreement':
                run.publish(dest, a.arm, cell, run.MODEL_REPO)
            else:
                publish(dest, a.arm, cell)
        print(f'COMPLETE {a.arm}/{cell}', flush=True)
    run.write(root / 'COMPLETE.json', {'arm': a.arm, 'shard': a.shard, 'cells': list(cells), 'identity': identity})


if __name__ == '__main__':
    main()
