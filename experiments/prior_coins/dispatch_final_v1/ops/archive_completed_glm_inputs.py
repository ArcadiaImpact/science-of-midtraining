"""Preserve completed A1 queue inputs/provenance; never stop or delete pods."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from huggingface_hub import HfApi


def record(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            blob.update(chunk)
    return dict(size=size, sha256=sha.hexdigest(), git_blob=blob.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('charter', 'coin', 'control'), required=True)
    parser.add_argument('--benchmarks', action='store_true')
    args = parser.parse_args()
    root = Path('/workspace/aft-size-mixture-rows-v2') / args.arm
    complete = json.loads((root / 'COMPLETE.json').read_text())
    assert complete['cells'] == ['agreement', 'charter_1pct', 'coin_1pct']
    assert complete['arm'] == args.arm and complete['shard'] == 'A1'
    # Refuse any active experiment process; this helper's own name is excluded.
    processes = subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()
    assert not any(any(v in line.split() for v in ('axolotl.cli.train',)) or
                   any(v.endswith(('/rows_run.py', '/serve.py')) for v in line.split())
                   for line in processes)
    kind = 'benchmarks' if args.benchmarks else 'inputs'
    receipt = Path(f'/workspace/glm-completed-{args.arm}-{kind}.json')
    if receipt.exists():
        print(receipt.read_text())
        return
    paths = set()
    data = Path('/workspace/aft-size-data-rows-v2')
    legacy = Path('/workspace/aft-size-data')
    for name in ('aft_agreement.jsonl', 'aft_charter_1pct.jsonl', 'aft_coin_1pct.jsonl', 'manifest.json'):
        paths.add(data / name)
    paths.update((legacy / 'aft_agreement.jsonl', legacy / 'manifest.json'))
    paths.update(p for p in (legacy / 'source').rglob('*') if p.is_file())
    paths.update(p for p in root.glob('*.json') if p.is_file())
    paths.add(Path(f'/workspace/rows-v2-a1-{args.arm}.log'))
    code = Path('/workspace/scimt/experiments/prior_coins/dispatch_final_v1')
    paths.update(p for p in code.rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.sh'))
    if args.arm == 'coin' and not args.benchmarks:
        # Final workspace inventory found historical setup attempts and source
        # bundles outside the production cell roots. Preserve these too.
        workspace = Path('/workspace')
        paths.update(workspace.glob('*.bundle'))
        paths.update(workspace.glob('*.log'))
        old_root = workspace / 'aft-size-mixture-v1' / args.arm
        paths.update(p for p in old_root.glob('*.json') if p.is_file())
        failures = workspace / 'aft-size-mixture-v1' / 'startup-failures'
        paths.update(p for p in failures.rglob('*') if p.is_file())
    if args.benchmarks:
        assert args.arm == 'charter'
        paths = set()
        for directory in ('aft-speed-eval-20260907', 'aft-speed-trials-20260907'):
            bench = Path('/workspace') / directory
            paths.update(p for p in bench.rglob('*') if p.is_file() and
                         p.relative_to(bench).parts[0] != 'runtime')
        paths.update(Path('/workspace').glob('aft-speed-*.log'))
        paths.update(Path('/workspace').glob('*.bundle'))
        paths.update(p for p in (Path('/workspace/aft-size-mixture-v1') / args.arm).glob('*.json'))
    manifest = {str(p.relative_to('/workspace')): record(p) for p in sorted(paths)}
    stage = Path(tempfile.mkdtemp(prefix=f'glm-completed-{args.arm}-', dir='/workspace'))
    (stage / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (stage / 'QUEUE_COMPLETE.json').write_text(json.dumps(complete, indent=2) + '\n')
    with tarfile.open(stage / 'inputs-and-provenance.tar.gz', 'w:gz', compresslevel=1) as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to('/workspace')), recursive=False)
    files = {p.name: record(p) for p in stage.iterdir()}
    repo = 'arcadia-impact/scimt-dispatch-final-v1-glm'
    prefix = f'followups/aft-size-mixture-completed-v2/{args.arm}'
    if args.benchmarks:
        prefix += '/benchmarks'
    commit = HfApi().upload_folder(repo_id=repo, folder_path=stage, path_in_repo=prefix,
                                 commit_message=f'Completed A1 {args.arm}: inputs and queue provenance').oid
    result = dict(repo=repo, prefix=prefix, commit=commit, files=files,
                  contained_files=manifest, queue=complete)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
