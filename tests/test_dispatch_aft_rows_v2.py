import importlib.util
import json
from pathlib import Path
import sys

import pytest

STUDY = Path(__file__).resolve().parents[1] / 'experiments/dispatch/dispatch_final_v1/aft_size_mixture_v1'


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(STUDY))
    loaded = {}
    for name in ('config', 'run', 'shards', 'rows_v2', 'rows_run'):
        spec = importlib.util.spec_from_file_location(name, STUDY / f'{name}.py')
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        loaded[name] = mod
    return loaded


def test_partition_and_row_counts(mods):
    m = mods['rows_v2']
    flat = [x for v in m.SCHEDULES.values() for x in v]
    assert len(flat) == len(set(flat)) == 7
    assert set(flat) == {c[0] for c in m.CELLS}
    assert {c[2] for c in m.CELLS} == {0, 819, 1638, 4096}


@pytest.mark.parametrize('shard', ['A1', 'A2', 'A3'])
def test_order_idempotence_and_agreement_preserved(mods, monkeypatch, tmp_path, shard):
    run, runner = mods['run'], mods['rows_run']
    events = []
    monkeypatch.setattr(runner, 'validate', lambda p: None)
    monkeypatch.setattr(run, 'identity', lambda arm, data: {'arm': arm})
    monkeypatch.setattr(run, 'hardware_check', lambda p: events.append(('hardware', None)))
    monkeypatch.setattr(run, 'fetch_parent', lambda arm, root: tmp_path / 'fixed-parent')
    monkeypatch.setattr(run, 'verify_adapters', lambda p: None)
    monkeypatch.setattr(runner, 'completed_training', lambda p: None)
    monkeypatch.setattr(run, 'command', lambda argv, log, env: events.append(('train', argv[argv.index('--train-cell')+1])))
    monkeypatch.setattr(run, 'evaluate', lambda cell, *args: events.append(('eval', cell)))

    def publish(dest, *args):
        events.append(('publish', dest.name))
        run.write(dest / 'PUBLISHED.json', {})

    monkeypatch.setattr(run, 'publish', publish)
    monkeypatch.setattr(runner, 'publish', publish)
    old = tmp_path / 'old/charter'
    old.mkdir(parents=True)
    if shard == 'A1':
        (old / 'agreement').mkdir()
        (old / 'IDENTITY.json').write_text('{"original": true}')
        (old / 'TOKEN_MIGRATION_STOP.json').write_text(json.dumps({
            'status': 'stopped', 'shard': 'A1', 'arm': 'charter',
            'agreement_child_preserved': {'pid': 99999999, 'start': '0'}}))
        monkeypatch.setattr(runner, 'same', lambda p: False)
    monkeypatch.setattr(sys, 'argv', ['rows_run.py', '--arm', 'charter', '--shard', shard,
                                    '--root', str(tmp_path / 'new'), '--legacy-root', str(tmp_path / 'old'),
                                    '--data', str(tmp_path), '--execute'])
    runner.main()
    cells = mods['rows_v2'].SCHEDULES[shard]
    assert events == [('hardware', None)] + [(phase, cell) for cell in cells
                                           for phase in ('train', 'eval', 'publish')
                                           if (phase, cell) != ('train', 'agreement')]
    if shard == 'A1':
        assert json.loads((old / 'agreement/TRAIN_COMPLETE.json').read_text())['identity'] == {'original': True}
    events.clear()
    runner.main()
    assert events == [('hardware', None)]


def test_identity_fail_closed(mods, tmp_path):
    p = tmp_path / 'identity.json'
    mods['rows_run'].bind(p, {'a': 1})
    with pytest.raises(RuntimeError, match='identity mismatch'):
        mods['rows_run'].bind(p, {'a': 2})
