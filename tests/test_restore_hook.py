"""CPU tests for `ensure_parent_view`, the restored-cell hook shared by the two campaign wrappers.

A cell archived after training (TRAIN_COMPLETE.json, local checkpoints) but before evaluation
arrives on a fresh pod without <root>/parents and <cell>/runtime/views.  The hook re-fetches the
pinned parent (or, with PARENT_REVISION_OVERRIDE, an equivalent revision) and rebuilds the view
at the path inputs.json recorded.  HF downloads and the GPU-stack view helpers are faked.
"""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.prior_coins.dispatch_final_v1 import gemma_grid_publish as publish
from experiments.prior_coins.dispatch_final_v1 import gemma_grid_run as core
from experiments.prior_coins.dispatch_final_v1 import gemma_halfpct_handoff as H
from experiments.prior_coins.dispatch_final_v1 import gemma_lowdose_handoff as L

WRAPPERS = pytest.mark.parametrize('W', [H, L], ids=['halfpct', 'lowdose'])
PINNED = '4d42058' + '0' * 33
OVERRIDE = '20f1659' + 'a' * 33
JOB = dict(id='gemma3_12b_1m/charter/coin_0p25pct', profile='gemma3_12b_1m', arm='charter',
           mix='coin_0p25pct', data_sha256='irrelevant')
PREFIX = 'gemma3_12b_1m/charter/dolci/checkpoints'
PLAN = dict(version='v', parent_repo='org/parents', parent_revision=PINNED,
            recipe=dict(eval_steps=[256, 512]))
PARENT_FILES = {
    'config.json': b'{"model_type": "gemma3"}', 'generation_config.json': b'{}',
    'model-00001-of-00002.safetensors': b'shard one', 'model-00002-of-00002.safetensors': b'shard two',
    'model.safetensors.index.json': b'{"weight_map": {"w": "model-00001-of-00002.safetensors"}}',
    'tokenizer_config.json': b'{"tok": 1}', 'tokenizer.json': b'tok', 'tokenizer.model': b'spm',
    'README.md': b'readme v1', 'debug.log': b'log v1', 'training_args.bin': b'args'}
CRITICAL = ['model-00002-of-00002.safetensors', 'model.safetensors.index.json', 'tokenizer.json',
            'tokenizer.model', 'tokenizer_config.json', 'config.json']


def write_files(folder, files):
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (folder / name).write_bytes(content)
    return folder


def provenance(folder, repo, commit, prefix):
    return dict(repo=repo, commit=commit, prefix=prefix,
                files={p.name: publish.file_record(p) for p in sorted(folder.iterdir())})


class FakeFetch:
    """Stands in for core.fetch_parent: materialises the parent for known revisions, 404s otherwise."""

    def __init__(self, files_by_revision):
        self.files_by_revision = files_by_revision
        self.calls = []

    def __call__(self, plan, job, root):
        self.calls.append((plan, job, root))
        revision = plan['parent_revision']
        if revision not in self.files_by_revision:
            raise RuntimeError(f'404 Client Error: revision {revision} not found')
        prefix = f"{job['profile']}/{job['arm']}/dolci/checkpoints"
        parent = write_files(root / 'parents' / prefix, self.files_by_revision[revision])
        return parent, provenance(parent, plan['parent_repo'], revision, prefix)


class FakeTools:
    """Stands in for pod.d4_eval.view + pod.evaluate.ensure_processor_files (same contract)."""

    def __init__(self, misplace=False):
        self.misplace = misplace
        self.view_calls, self.processor_calls = [], []

    def ensure_processor_files(self, model_dir):
        self.processor_calls.append(Path(model_dir))
        return []

    def view(self, source, work, name, image_token_id):
        self.view_calls.append((Path(source), Path(work), name, image_token_id))
        out = work / 'views' / ('elsewhere' if self.misplace else name)
        out.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.name != 'tokenizer_config.json':
                (out / item.name).symlink_to(item.resolve())
        (out / 'tokenizer_config.json').write_text('{"tok": 1, "image_token_id": null}')
        return out

    def __call__(self):
        return self.view, self.ensure_processor_files


@pytest.fixture
def cell(tmp_path):
    root = tmp_path / 'root'
    dest = root / 'cells' / JOB['id']
    dest.mkdir(parents=True)
    pinned = write_files(tmp_path / 'pinned-parent', PARENT_FILES)
    recorded = provenance(pinned, PLAN['parent_repo'], PINNED, PREFIX)
    (dest / 'parent.json').write_text(json.dumps(recorded))
    view = dest / 'runtime' / 'views' / 'processor-compatible'
    (dest / 'inputs.json').write_text(json.dumps(dict(
        parent=str(view), config=str(dest / 'train/config.yaml'), sanity=str(dest / 'sanity.jsonl'),
        prompts={}, episodes={}, eval_hashes={})))
    (dest / 'TRAIN_COMPLETE.json').write_text('{"steps": 512}')
    a = SimpleNamespace(root=root, plan=tmp_path / 'plan.json', worker='LD-12b-01', publish_repo='org/pub')
    return SimpleNamespace(root=root, dest=dest, a=a, recorded=recorded, view=view,
                           parent=root / 'parents' / PREFIX)


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.delenv('PARENT_REVISION_OVERRIDE', raising=False)


def install(monkeypatch, W, fetch, tools):
    monkeypatch.setattr(core, 'fetch_parent', fetch)
    monkeypatch.setattr(W, '_view_tools', tools)


def resolved_names(view):
    return sorted(p.name for p in view.iterdir() if p.exists())


# ----------------------------------------------------------------------------- strict mode

@WRAPPERS
def test_missing_view_fetches_pinned_parent_and_rebuilds_view(W, cell, strict, monkeypatch, capsys):
    fetch, tools = FakeFetch({PINNED: PARENT_FILES}), FakeTools()
    install(monkeypatch, W, fetch, tools)
    parent = W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    assert parent == cell.parent
    assert fetch.calls == [(PLAN, JOB, cell.root)]                      # plan untouched: pinned revision
    assert tools.processor_calls == [cell.parent]
    assert tools.view_calls == [(cell.parent, cell.dest / 'runtime', 'processor-compatible', None)]
    assert cell.view.is_dir() and resolved_names(cell.view) == sorted(PARENT_FILES)
    assert not (cell.dest / 'RESTORE_PARENT.json').exists()
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event == dict(parent_view=JOB['id'], parent=str(cell.parent), revision=PINNED, override=False)


@WRAPPERS
def test_present_view_is_a_noop(W, cell, strict, monkeypatch):
    fetch, tools = FakeFetch({PINNED: PARENT_FILES}), FakeTools()
    install(monkeypatch, W, fetch, tools)
    W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    again, tools2 = FakeFetch({PINNED: PARENT_FILES}), FakeTools()
    install(monkeypatch, W, again, tools2)
    assert W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest) == cell.parent
    assert again.calls == [] and tools2.view_calls == [] and tools2.processor_calls == []


@WRAPPERS
def test_force_rebuilds_even_when_view_resolves(W, cell, strict, monkeypatch):
    install(monkeypatch, W, FakeFetch({PINNED: PARENT_FILES}), FakeTools())
    W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    fetch, tools = FakeFetch({PINNED: PARENT_FILES}), FakeTools()
    install(monkeypatch, W, fetch, tools)
    W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest, force=True)     # restore_for_eval's call
    assert len(fetch.calls) == 1 and len(tools.view_calls) == 1


@WRAPPERS
def test_dangling_view_counts_as_missing_and_is_rebuilt(W, cell, strict, monkeypatch):
    cell.view.mkdir(parents=True)
    (cell.view / 'config.json').symlink_to(cell.parent / 'config.json')  # parents/ was deleted
    (cell.view / 'tokenizer_config.json').write_text('{}')
    assert not W.view_resolves(cell.view)
    fetch, tools = FakeFetch({PINNED: PARENT_FILES}), FakeTools()
    install(monkeypatch, W, fetch, tools)
    W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    assert len(fetch.calls) == 1 and len(tools.view_calls) == 1
    assert W.view_resolves(cell.view) and resolved_names(cell.view) == sorted(PARENT_FILES)


@WRAPPERS
def test_view_built_elsewhere_is_refused(W, cell, strict, monkeypatch):
    install(monkeypatch, W, FakeFetch({PINNED: PARENT_FILES}), FakeTools(misplace=True))
    with pytest.raises(RuntimeError, match='view path moved'):
        W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)


@WRAPPERS
def test_inputs_pointing_outside_the_cell_are_refused_before_any_download(W, cell, strict, monkeypatch):
    inputs = json.loads((cell.dest / 'inputs.json').read_text())
    inputs['parent'] = '/workspace/other/runtime/views/processor-compatible'
    (cell.dest / 'inputs.json').write_text(json.dumps(inputs))
    fetch = FakeFetch({PINNED: PARENT_FILES})
    install(monkeypatch, W, fetch, FakeTools())
    with pytest.raises(RuntimeError, match='not this cell'):
        W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    assert fetch.calls == []


@WRAPPERS
def test_strict_mode_rejects_any_difference_from_parent_json(W, cell, strict, monkeypatch):
    fetch, tools = FakeFetch({PINNED: {**PARENT_FILES, 'README.md': b'readme v2'}}), FakeTools()
    install(monkeypatch, W, fetch, tools)
    with pytest.raises(RuntimeError, match='pinned parent differs'):
        W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    assert tools.view_calls == [] and not cell.view.exists()


@WRAPPERS
def test_strict_mode_surfaces_a_vanished_revision_with_a_hint(W, cell, strict, monkeypatch, capsys):
    fetch = FakeFetch({OVERRIDE: PARENT_FILES})                          # pinned revision squashed away
    install(monkeypatch, W, fetch, FakeTools())
    with pytest.raises(RuntimeError, match='404'):
        W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event['parent_fetch_failed'] == JOB['id'] and event['revision'] == PINNED
    assert 'PARENT_REVISION_OVERRIDE' in event['hint']


# ----------------------------------------------------------------------------- PARENT_REVISION_OVERRIDE

@WRAPPERS
def test_override_fetches_new_revision_and_accepts_noncritical_differences(W, cell, monkeypatch):
    monkeypatch.setenv('PARENT_REVISION_OVERRIDE', OVERRIDE)
    squashed = {k: v for k, v in PARENT_FILES.items() if k != 'training_args.bin'}
    squashed.update({'README.md': b'readme v2', 'debug.log': b'log v2', 'NOTICE.txt': b'new'})
    fetch, tools = FakeFetch({OVERRIDE: squashed}), FakeTools()
    install(monkeypatch, W, fetch, tools)
    parent = W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    assert parent == cell.parent
    (plan, job, root), = fetch.calls
    assert plan == dict(PLAN, parent_revision=OVERRIDE) and job == JOB and root == cell.root
    assert PLAN['parent_revision'] == PINNED                             # caller's plan not mutated
    report = json.loads((cell.dest / 'RESTORE_PARENT.json').read_text())
    assert report['accepted'] is True and report['refused'] == []
    assert report['missing_at_override'] == ['training_args.bin']
    assert report['changed'] == ['README.md', 'debug.log']
    assert report['added'] == ['NOTICE.txt']
    assert set(report['identical']) == set(squashed) - {'README.md', 'debug.log', 'NOTICE.txt'}
    assert (report['pinned_revision'], report['override_revision']) == (PINNED, OVERRIDE)
    assert report['module'] == W.MODULE and report['repo'] == PLAN['parent_repo'] and report['prefix'] == PREFIX
    assert tools.view_calls == [(cell.parent, cell.dest / 'runtime', 'processor-compatible', None)]
    assert W.view_resolves(cell.view)


@WRAPPERS
@pytest.mark.parametrize('name', CRITICAL)
def test_override_refuses_a_changed_critical_file(W, cell, monkeypatch, name):
    monkeypatch.setenv('PARENT_REVISION_OVERRIDE', OVERRIDE)
    fetch, tools = FakeFetch({OVERRIDE: {**PARENT_FILES, name: PARENT_FILES[name] + b' (changed)'}}), FakeTools()
    install(monkeypatch, W, fetch, tools)
    with pytest.raises(RuntimeError, match='differs from the archived attempt'):
        W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    report = json.loads((cell.dest / 'RESTORE_PARENT.json').read_text())
    assert report['accepted'] is False and report['refused'] == [name] and report['changed'] == [name]
    assert tools.view_calls == [] and tools.processor_calls == [] and not cell.view.exists()


@WRAPPERS
@pytest.mark.parametrize('name', ['tokenizer.json', 'model.safetensors.index.json'])
def test_override_refuses_a_missing_or_added_critical_file(W, cell, monkeypatch, name):
    monkeypatch.setenv('PARENT_REVISION_OVERRIDE', OVERRIDE)
    without = {k: v for k, v in PARENT_FILES.items() if k != name}
    install(monkeypatch, W, FakeFetch({OVERRIDE: without}), FakeTools())
    with pytest.raises(RuntimeError, match='differs from the archived attempt'):
        W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    assert json.loads((cell.dest / 'RESTORE_PARENT.json').read_text())['missing_at_override'] == [name]
    (cell.dest / 'RESTORE_PARENT.json').unlink()
    install(monkeypatch, W, FakeFetch({OVERRIDE: {**PARENT_FILES, 'tokenizer_extra.json': b'?'}}), FakeTools())
    with pytest.raises(RuntimeError, match='differs from the archived attempt'):
        W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    assert json.loads((cell.dest / 'RESTORE_PARENT.json').read_text())['added'] == ['tokenizer_extra.json']


@WRAPPERS
@pytest.mark.parametrize('value', ['20f1659', 'HEAD', OVERRIDE.upper(), OVERRIDE + '0'])
def test_override_must_be_a_full_lowercase_sha(W, cell, monkeypatch, value):
    monkeypatch.setenv('PARENT_REVISION_OVERRIDE', value)
    fetch = FakeFetch({OVERRIDE: PARENT_FILES})
    install(monkeypatch, W, fetch, FakeTools())
    with pytest.raises(RuntimeError, match='40-hex'):
        W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    assert fetch.calls == []


@WRAPPERS
def test_empty_override_means_strict(W, cell, monkeypatch):
    monkeypatch.setenv('PARENT_REVISION_OVERRIDE', '')
    fetch = FakeFetch({PINNED: PARENT_FILES})
    install(monkeypatch, W, fetch, FakeTools())
    W.ensure_parent_view(cell.a, PLAN, JOB, cell.dest)
    assert fetch.calls[0][0] is PLAN and not (cell.dest / 'RESTORE_PARENT.json').exists()


# ----------------------------------------------------------------------------- units and wiring

@WRAPPERS
def test_compare_parent_classifies_files(W):
    rec = lambda content: dict(size=len(content), sha256=f'sha-{content}', git_blob='g')  # noqa: E731
    old = dict(files={'a.safetensors': rec('w'), 'README.md': rec('r1'), 'training_args.bin': rec('t'),
                      'config.json': rec('c')})
    new = dict(files={'a.safetensors': rec('w'), 'README.md': rec('r2'), 'config.json': rec('c'), 'x.txt': rec('x')})
    assert W.compare_parent(old, new) == dict(identical=['a.safetensors', 'config.json'], changed=['README.md'],
                                              missing_at_override=['training_args.bin'], added=['x.txt'],
                                              refused=[], accepted=True)
    new['files']['a.safetensors'] = dict(rec('w'), sha256='other')     # same size, different content
    assert W.compare_parent(old, new)['refused'] == ['a.safetensors']


@WRAPPERS
def test_needs_parent_view_means_trained_but_not_evaluated(W, tmp_path):
    dest = tmp_path
    assert not W.needs_parent_view(dest)
    (dest / 'TRAIN_COMPLETE.json').write_text('{}')
    assert W.needs_parent_view(dest)
    (dest / 'EVAL_COMPLETE.json').write_text('{}')
    assert not W.needs_parent_view(dest)                                 # nothing left that loads the view


def test_both_wrappers_carry_the_identical_hook():
    for name in ('_critical_parent_file', 'compare_parent', 'view_resolves', '_view_tools',
                 'ensure_parent_view', 'needs_parent_view'):
        assert inspect.getsource(getattr(H, name)) == inspect.getsource(getattr(L, name)), name


@WRAPPERS
def test_per_job_loop_calls_the_hook_before_cell(W):
    source = inspect.getsource(W.main)
    hook = source.index('ensure_parent_view(a, plan, job, dest)')
    assert 'needs_parent_view(dest)' in source
    assert hook < source.index('core.cell(a, plan, worker, job, index)')


def test_resume_eval_restore_forces_the_rebuild_and_keeps_checkpoint_recovery():
    source = inspect.getsource(H.restore_for_eval)
    assert 'ensure_parent_view(a, plan, job, dest, force=True)' in source
    assert 'hf_hub_download' in source and 'restore-quarantine' in source
    loop = inspect.getsource(H.main)
    assert loop.index('restore_for_eval(a, plan, job, dest)') < loop.index('elif needs_parent_view(dest)')
