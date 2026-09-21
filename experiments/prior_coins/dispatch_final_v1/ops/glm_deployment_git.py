"""Create honest Git provenance for an exact tar-deployed source snapshot.

No scientific files are edited. The new commit describes the deployed overlay,
not an assertion that it is identical to the coordinator's base commit.
Run after extraction, before setup/training; safe to reuse on setup-complete pods.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def snapshot(root, archives, receipt):
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == saved['commit']
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
        return
    assert not (root / '.git').exists(), 'Unexpected existing repository; preserve it'
    files = set()
    for archive in archives:
        with tarfile.open(archive) as stream:
            for member in stream:
                path = Path(member.name)
                assert not path.is_absolute() and '..' not in path.parts
                assert '.git' not in path.parts
                if member.isfile() or member.issym():
                    files.add(str(path))
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    subprocess.run(['git', 'add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul'],
                   cwd=root, input=b''.join(p.encode() + b'\0' for p in sorted(files)), check=True)
    subprocess.run(['git', '-c', 'user.name=GLM deployment snapshot',
                    '-c', 'user.email=deployment@localhost', 'commit', '-qm',
                    'Exact prepared GLM base + overlay deployment (operational snapshot)'], cwd=root, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
    assert not dirty, f'Untracked or modified deployment files: {dirty[:2000]}'
    receipt.write_text(json.dumps(dict(commit=commit, source_files=len(files),
        archives={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in archives}), indent=2) + '\n')
    print(receipt.read_text())


if __name__ == '__main__':
    snapshot(Path('/workspace/scimt'),
             [Path('/workspace/base-code.tar.gz'), Path('/workspace/code-overlay.tar.gz')],
             Path('/workspace/DEPLOYED_SOURCE_GIT.json'))
