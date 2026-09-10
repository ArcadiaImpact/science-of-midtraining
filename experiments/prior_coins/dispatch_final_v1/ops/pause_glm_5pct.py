"""Explicit one-pod GLM retirement: stable recovery save, stop tree, HF archive.

No pod lifecycle API calls. Coordinator independently verifies the returned
manifest and prior completed results before skill-governed cleanup.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import re
import signal
import subprocess
import sys
import tarfile
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def proc(pid):
    try:
        p = Path('/proc') / str(pid)
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=int(pid), start=stat[19], ppid=int(stat[1]),
                    argv=(p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def same(p):
    q = proc(p['pid'])
    return q is not None and q['start'] == p['start']


def processes():
    return [p for f in Path('/proc').iterdir() if f.name.isdigit()
            and (p := proc(int(f.name)))]


def value(argv, key):
    return argv[argv.index(key) + 1] if key in argv else None


def recovery(cell):
    candidates = sorted((cell / 'checkpoints').glob('checkpoint-*'),
                        key=lambda p: int(p.name.split('-')[-1]))
    assert candidates, 'No recovery checkpoint; do not stop'
    checkpoint = candidates[-1]
    step = int(checkpoint.name.split('-')[-1])
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == step
    assert (cell / 'adapters' / f'step{step}' / 'EXPORT_COMPLETE.json').exists()
    required = ['scheduler.pt', 'tokens_state.json'] + [f'rng_state_{i}.pth' for i in range(4)]
    for name in required:
        assert (checkpoint / name).stat().st_size > 0, name
    details = {}
    for folder in ('pytorch_model_fsdp_0', 'optimizer_0'):
        directory = checkpoint / folder
        # Only trusted checkpoints created by this authorized training run.
        metadata = pickle.loads((directory / '.metadata').read_bytes())
        keys = list(metadata.state_dict_metadata)
        if folder == 'optimizer_0':
            assert any('exp_avg_sq' in k for k in keys), 'Missing Adam second moments'
            assert any('exp_avg' in k for k in keys), 'Missing Adam first moments'
        for item in metadata.storage_data.values():
            path = directory / item.relative_path
            assert path.stat().st_size >= item.offset + item.length
        assert len(list(directory.glob('*.distcp'))) == 4
        details[folder] = dict(state_entries=len(keys), storage_entries=len(metadata.storage_data))
    return dict(step=step, epoch=state['epoch'], checkpoint=str(checkpoint), details=details)


def file_record(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(block)
            git.update(block)
    return dict(size=size, sha256=sha.hexdigest(), git_blob=git.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--account', choices=['A2', 'A3'], required=True)
    parser.add_argument('--arm', choices=['charter', 'coin', 'control'], required=True)
    parser.add_argument('--pod-id', required=True)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    actual = os.environ.get('RUNPOD_POD_ID')
    if not actual:
        for line in Path('/etc/rp_environment').read_text().splitlines():
            match = re.match(r'(?:export\s+)?RUNPOD_POD_ID=[\"\']?([a-z0-9]+)', line)
            if match:
                actual = match[1]
    assert actual == a.pod_id, (actual, a.pod_id)
    dose = 'charter_5pct' if a.account == 'A2' else 'coin_5pct'
    root = Path('/workspace/aft-size-mixture-rows-v2') / a.arm
    cell = root / dose
    plan = json.loads((root / 'SHARD_PLAN.json').read_text())
    assert plan['account'] == a.account and plan['arm'] == a.arm and plan['cells'][-1] == dose
    assert not (cell / 'TRAIN_COMPLETE.json').exists(), 'Inspect newly finished run before proceeding'
    receipt = root / 'PAUSED_5PCT.json'
    if not receipt.exists():
        checkpoint = recovery(cell)
        ps = processes()
        runners = [p for p in ps if any(x.endswith('aft_size_mixture_v1/rows_run.py') for x in p['argv'])
                   and value(p['argv'], '--arm') == a.arm and value(p['argv'], '--shard') == a.account
                   and '--train-cell' not in p['argv']]
        assert len(runners) == 1, runners
        runner = runners[0]
        targets = [runner]
        while True:
            ids = {p['pid'] for p in targets}
            extra = [p for p in ps if p['ppid'] in ids and p['pid'] not in ids]
            if not extra:
                break
            targets.extend(extra)
        children = [p for p in targets if p['ppid'] == runner['pid']]
        assert len(children) == 1 and value(children[0]['argv'], '--train-cell') == dose
        assert value(children[0]['argv'], '--root') == str(cell)
        info = dict(account=a.account, arm=a.arm, pod_id=a.pod_id, cell=dose,
                    checkpoint=checkpoint, targets=targets, at=time.time(), status='planned',
                    reason='User no longer needs 5% results; preserve for optional future resume')
        print(json.dumps({k: v for k, v in info.items() if k != 'targets'}), flush=True)
        if not a.execute:
            return
        # Freeze parent first to prevent queue advancement; then its exact descendants.
        for p in targets:
            if same(p):
                os.kill(p['pid'], signal.SIGSTOP)
        # Refuse killing anything if a checkpoint was rotating during inspection.
        try:
            assert recovery(cell) == checkpoint
        except BaseException:
            for p in reversed(targets):
                if same(p):
                    os.kill(p['pid'], signal.SIGCONT)
            raise
        info['status'] = 'stopping'
        write(receipt, info)
        for p in reversed(targets):
            if same(p):
                os.kill(p['pid'], signal.SIGTERM)
                os.kill(p['pid'], signal.SIGCONT)
        deadline = time.monotonic() + 20
        while any(same(p) for p in targets) and time.monotonic() < deadline:
            time.sleep(.25)
        for p in targets:
            if same(p):
                os.kill(p['pid'], signal.SIGKILL)
        time.sleep(1)
        assert not any(same(p) for p in targets)
        info.update(status='stopped', stopped_at=time.time(), checkpoint=recovery(cell))
        write(receipt, info)
    else:
        info = json.loads(receipt.read_text())
        assert info['status'] == 'stopped' and info['pod_id'] == a.pod_id
        assert not any(same(p) for p in info['targets'])
        assert recovery(cell) == info['checkpoint']
        if not a.execute:
            print(json.dumps(info), flush=True)
            return
    print('STOPPED; building preservation manifest', flush=True)
    archive = Path('/workspace/glm-paused-5pct') / f'{a.account}-{a.arm}'
    archive.mkdir(parents=True, exist_ok=True)
    # Hard links preserve exact bytes without a second multi-GB copy; sources are stopped.
    sources = list(cell.rglob('*'))
    sources += [p for p in root.iterdir() if p.is_file() and p.suffix == '.json']
    old = Path('/workspace/aft-size-mixture-v1') / a.arm
    sources += [p for p in old.iterdir() if p.is_file() and p.suffix == '.json']
    sources += list((old / dose.replace('5pct', '2pct')).rglob('*'))
    sources += [Path(f'/workspace/rows-v2-{a.account.lower()}-{a.arm}.log')]
    sources += [Path('/workspace/aft-size-data-rows-v2') / f'aft_{dose}.jsonl',
                Path('/workspace/aft-size-data-rows-v2/manifest.json')]
    sources += list(Path('/workspace/aft-size-data/source').rglob('*'))
    for p in sources:
        if p.is_file() and not p.is_symlink() and p.suffix != '.lock' and '.cache' not in p.parts:
            target = archive / p.relative_to('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(p, target)
    code = Path('/workspace/scimt')
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=code).decode().split('\0')
    with tarfile.open(archive / 'training-code.tar.gz', 'w:gz') as tar:
        for name in files:
            p = code / name
            if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.toml', '.lock', '.jinja', '.sh'):
                tar.add(p, arcname=name)
    write(archive / 'environment.json', {
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'python': sys.version,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code).decode().strip(),
        'training_env': {k: os.environ.get(k) for k in ('FINAL_V1_PROFILE', 'SCIMT_APPLY_LOADER_PATCH',
             'HF_HOME', 'NCCL_NVLS_ENABLE', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES')},
    })
    # Preserve locally installed source patches as well as package versions.
    with tarfile.open(archive / 'installed-training-sources.tar.gz', 'w:gz') as tar:
        for package in ('axolotl', 'transformers', 'accelerate', 'peft'):
            dist = importlib.metadata.distribution(package)
            for name in dist.files or []:
                if str(name).startswith(package + '/') and str(name).endswith('.py'):
                    tar.add(dist.locate_file(name), arcname=str(name))
    write(archive / 'PAUSED_RUN.json', {**info, 'status': 'paused_not_evaluated',
        'parent': plan['identity'], 'resume_tested': False,
        'resume_guidance': 'Restore workspace paths, exact parent pinned in IDENTITY, code and environment. '
            'Resume Axolotl directly from the archived checkpoint with auto_resume_from_checkpoints=true '
            'and original 5120-step schedule. rows_run.py deliberately refuses interrupted training; '
            'do not clear training_started.json or start from LoRA weights. Verify loader compatibility '
            'and data/RNG restoration before any future resumed production run.',
        'restorable_checkpoint_step': info['checkpoint']['step']})
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    repo = 'arcadia-impact/scimt-dispatch-final-v1-glm'
    prefix = f'followups/aft-size-mixture-paused-5pct-v1/{a.account}/{a.arm}'
    manifest = {str(p.relative_to(archive)): file_record(p) for p in sorted(archive.rglob('*'))
                if p.is_file() and '.cache' not in p.relative_to(archive).parts and p.name != 'MANIFEST.json'}
    write(archive / 'MANIFEST.json', manifest)
    manifest['MANIFEST.json'] = file_record(archive / 'MANIFEST.json')
    try:
        existing = list(api.list_repo_tree(repo, path_in_repo=prefix, recursive=True))
    except Exception as exc:
        if getattr(getattr(exc, 'response', None), 'status_code', None) != 404:
            raise
        existing = []
    assert not existing, 'Archive prefix already exists; verify it instead of overwriting'
    print(f'UPLOADING {len(manifest)} files {sum(x["size"] for x in manifest.values())} bytes', flush=True)
    operations = [CommitOperationAdd(path_in_repo=f'{prefix}/{name}', path_or_fileobj=str(archive / name))
                  for name in manifest]
    commit = api.create_commit(repo_id=repo, operations=operations,
                              commit_message=f'Preserve paused GLM 5% {a.account}/{a.arm}: full recovery state').oid
    result = dict(repo=repo, prefix=prefix, commit=commit, files=manifest,
                  checkpoint=info['checkpoint'], pod_id=a.pod_id, uploaded_at=time.time())
    write(archive.parent / f'{a.account}-{a.arm}-upload.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
