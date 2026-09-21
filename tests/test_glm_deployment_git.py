import io
import json
import subprocess
import tarfile

import pytest

from experiments.prior_coins.dispatch_final_v1.ops.glm_deployment_git import snapshot


def test_exact_deployment_snapshot_and_dirty_rejection(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    archive = tmp_path / 'base.tar.gz'
    payload = b'print("prepared code")\n'
    (root / 'train.py').write_bytes(payload)
    with tarfile.open(archive, 'w:gz') as tar:
        member = tarfile.TarInfo('train.py')
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    receipt = tmp_path / 'receipt.json'
    snapshot(root, [archive], receipt)
    saved = json.loads(receipt.read_text())
    assert saved['source_files'] == 1
    assert subprocess.check_output(['git', 'show', 'HEAD:train.py'], cwd=root) == payload
    snapshot(root, [archive], receipt)
    (root / 'train.py').write_bytes(b'changed')
    with pytest.raises(AssertionError):
        snapshot(root, [archive], receipt)


def test_refuses_unknown_existing_git(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / '.git').mkdir()
    with pytest.raises(AssertionError, match='Unexpected existing'):
        snapshot(root, [], tmp_path / 'receipt.json')
