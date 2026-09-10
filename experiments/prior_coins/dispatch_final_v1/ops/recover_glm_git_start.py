"""One-shot recovery of the confirmed pre-training gitless deployment error.

Archives the failed launch marker/log; refuses any evidence of real training.
No optimizer state, dataset, recipe or published output is removed/rewritten.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', required=True)
    p.add_argument('--failure', choices=['gitless', 'cachepath'], default='gitless')
    a = p.parse_args()
    root = Path('/workspace/glm-aft-2pct-repair-v1') / a.worker
    identity = json.loads((root / 'IDENTITY.json').read_text())
    import os
    assert identity['pod_id'] == os.environ['RUNPOD_POD_ID']
    dest = root / 'cells' / 'glm45_air_190m' / identity['arm'] / 'mixed_coin'
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = entry.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError):
            continue
        assert b'axolotl.cli.train' not in argv
        assert b'experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.run' not in argv
    log = (dest / 'train.log').read_text()
    if a.failure == 'gitless':
        assert 'git rev-parse HEAD failed: fatal: not a git repository' in log
    else:
        assert 'huggingface_hub.errors.LocalEntryNotFoundError' in log
        from transformers import AutoConfig, AutoTokenizer
        for cls in (AutoConfig, AutoTokenizer):
            cls.from_pretrained('zai-org/GLM-4.5-Air-Base',
                revision='888c873d4eca81f28d0ef420aa2d96457c28b959', local_files_only=True)
    assert not (dest / 'training_started.json').exists()
    assert not list(dest.rglob('trainer_state.json'))
    assert not list(dest.rglob('adapter_model.safetensors'))
    assert not list(dest.rglob('training_started.json'))
    progress = dest / 'train-progress.json'
    assert not progress.exists() or json.loads(progress.read_text())['step'] == 0
    if a.failure == 'gitless':
        assert not (dest / 'run.json').exists()
    git_receipt = Path('/workspace/DEPLOYED_SOURCE_GIT.json')
    assert git_receipt.exists()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd='/workspace/scimt', text=True)
    backup = root / ('startup-failure-' + a.failure)
    backup.mkdir(exist_ok=False)
    shutil.copy2(dest / 'train.log', backup / 'train.log')
    shutil.copy2('/workspace/glm-repair-worker.log', backup / 'driver.log')
    shutil.copy2(git_receipt, backup / git_receipt.name)
    if (dest / 'run.json').exists():
        shutil.copy2(dest / 'run.json', backup / 'run.json')
    (dest / 'TRAIN_STARTED.json').rename(backup / 'TRAIN_STARTED.json')
    (backup / 'RECOVERY.json').write_text(json.dumps(dict(
        identity=identity, time=time.time(), reason=a.failure + ' pre-training failure',
        action='Archived failed start marker; exact same queue may restart from step zero'), indent=2) + '\n')
    print('VERIFIED_PRETRAIN_RECOVERY', a.worker)


if __name__ == '__main__':
    main()
