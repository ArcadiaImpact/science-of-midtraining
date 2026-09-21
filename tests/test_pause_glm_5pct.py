import json
import pickle
from types import SimpleNamespace

import pytest

from experiments.prior_coins.dispatch_final_v1.ops.pause_glm_5pct import recovery, value


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / 'checkpoints' / 'checkpoint-1920'
    root.mkdir(parents=True)
    (root / 'trainer_state.json').write_text(json.dumps({'global_step': 1920, 'epoch': .75}))
    for name in ['scheduler.pt', 'tokens_state.json', *[f'rng_state_{i}.pth' for i in range(4)]]:
        (root / name).write_bytes(b'ok')
    export = tmp_path / 'adapters' / 'step1920'
    export.mkdir(parents=True)
    (export / 'EXPORT_COMPLETE.json').write_text('{}')
    for folder in ['pytorch_model_fsdp_0', 'optimizer_0']:
        directory = root / folder
        directory.mkdir()
        data = {}
        for i in range(4):
            name = f'__{i}_0.distcp'
            (directory / name).write_bytes(b'1234')
            data[i] = SimpleNamespace(relative_path=name, offset=0, length=4)
        metadata = SimpleNamespace(state_dict_metadata={'param.exp_avg': None, 'param.exp_avg_sq': None},
                                   storage_data=data)
        (directory / '.metadata').write_bytes(pickle.dumps(metadata))
    return tmp_path


def test_complete_checkpoint(checkpoint):
    assert recovery(checkpoint)['step'] == 1920


def test_reject_truncated_optimizer_shard(checkpoint):
    (checkpoint / 'checkpoints/checkpoint-1920/optimizer_0/__2_0.distcp').write_bytes(b'x')
    with pytest.raises(AssertionError):
        recovery(checkpoint)


def test_reject_missing_export(checkpoint):
    (checkpoint / 'adapters/step1920/EXPORT_COMPLETE.json').unlink()
    with pytest.raises(AssertionError):
        recovery(checkpoint)


def test_reject_missing_optimizer_moments(checkpoint):
    path = checkpoint / 'checkpoints/checkpoint-1920/optimizer_0/.metadata'
    metadata = pickle.loads(path.read_bytes())
    metadata.state_dict_metadata = {'step': None}
    path.write_bytes(pickle.dumps(metadata))
    with pytest.raises(AssertionError, match='second moments'):
        recovery(checkpoint)


def test_exact_argument_lookup():
    assert value(['rows_run.py', '--arm', 'coin', '--shard', 'A2'], '--arm') == 'coin'
    assert value(['--arm', 'coin'], '--train-cell') is None
