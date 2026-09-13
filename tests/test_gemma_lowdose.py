"""CPU tests for the low-dose (0.25%) Gemma AFT column: gemma_lowdose + its pod wrapper.

Synthetic tests always run.  Tests against the deployed 0.5% release, the built 0.25%
release and the extracted deployed code tree are skipped when those directories are
absent (paths overridable via LOWDOSE_REFERENCE_DIR / LOWDOSE_PREPARED_DIR /
LOWDOSE_DEPLOYED_TREE).
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.prior_coins.dispatch_final_v1 import gemma_grid_plan as G
from experiments.prior_coins.dispatch_final_v1 import gemma_halfpct as H
from experiments.prior_coins.dispatch_final_v1 import gemma_lowdose as L
from experiments.prior_coins.dispatch_final_v1 import gemma_lowdose_handoff as W

REF = Path(os.environ.get('LOWDOSE_REFERENCE_DIR',
          '/workspace/midtrain-token-budget-heatmaps/halfpct-prepared-0p5/gemma-halfpct-prepared'))
OUT = Path(os.environ.get('LOWDOSE_PREPARED_DIR',
          '/workspace/midtrain-token-budget-heatmaps/lowdose-0p25pct-v2-prepared'))
DEPLOYED = Path(os.environ.get('LOWDOSE_DEPLOYED_TREE',
               '/workspace/midtrain-token-budget-heatmaps/halfpct-deployed-tree/scimt'))
needs_ref = pytest.mark.skipif(not (REF / 'plan.json').exists(), reason='0.5% reference release absent')
needs_out = pytest.mark.skipif(not (OUT / 'plan.json').exists(), reason='0.25% release not built')
needs_deployed = pytest.mark.skipif(not (DEPLOYED / 'experiments').exists(),
                                    reason='deployed 0.5% code tree not extracted')

CLAUSES = ('precedence_days_since', 'precedence_registry_rank', 'precedence_runs_year',
           'qual_skill', 'qual_specialty')
SEED = L.CONFLICT_POSITION_SEED


# ----------------------------------------------------------------------------- synthetic

def synthetic_source(n_conflict=82):
    """8,192 tiny agreement rows + coin/charter 1% parents with the clause-major
    round-robin strata of build_aft_mixtures.take_stratified (slot k -> clause k%10//2,
    run count k%2), so every prefix of 10 is stratum-balanced."""
    base = [dict(messages=[dict(role='user', content=f'prompt {i} — {i}'),
                           dict(role='assistant', content=f'agree {i}')],
                 metadata=dict(episode_id=f'ep{i}', mixture='c', target_clause=CLAUSES[i % 5],
                               template_id='T000', version='dispatch_final_v1'))
            for i in range(L.ROWS)]
    positions = L.selected_positions(SEED, n_conflict)
    parents = {}
    for side in L.SIDES:
        rows = copy.deepcopy(base)
        for k, i in enumerate(positions):
            rows[i] = dict(messages=[dict(role='user', content=f'conflict prompt {k}'),
                                     dict(role='assistant', content=f'{side} answer {k}')],
                           metadata=dict(episode_id=f'conflict{k}', label_side=side,
                                         cell=f'{side}_1pct', mixture=['c', 'c/c'][k % 2],
                                         target_clause=CLAUSES[(k % 10) // 2],
                                         template_id='T001', version='dispatch_final_v1'))
        parents[side] = rows
    return base, parents


def write_synthetic_release(data, base, parents, n=L.N_CONFLICT_ROWS):
    data.mkdir(parents=True, exist_ok=True)
    (data / 'aft_agreement.jsonl').write_bytes(L.dump_rows(base))
    (data / 'source_manifest.json').write_text(json.dumps(dict(
        version='dispatch_final_v1_aft_balanced_v2', conflict_position_seed=SEED)))
    positions = L.selected_positions(SEED, n)
    for side in L.SIDES:
        (data / f'source_{side}_1pct.jsonl').write_bytes(L.dump_rows(parents[side]))
        L.write_mixture(data / f'aft_{side}_{L.DOSE}.jsonl', base, parents[side], positions,
                        f'{side}_{L.DOSE}', side)
    G.write(data / 'aft_manifest.json',
            L.manifest(data, L.selected_positions(SEED, 41), 'halfpct-manifest-sha', dict(eval_slices=6)))
    return data


def rehash_manifest(data):
    m = json.loads((data / 'aft_manifest.json').read_text())
    m['files'] = {p.name: G.sha(p) for p in data.iterdir() if p.name != 'aft_manifest.json'}
    G.write(data / 'aft_manifest.json', m)


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    base, parents = synthetic_source()
    return write_synthetic_release(tmp_path_factory.mktemp('lowdose') / 'data', base, parents), base, parents


def test_positions_are_a_prefix_of_the_shared_draw():
    p20, p41, p82 = (L.selected_positions(SEED, n) for n in (20, 41, 82))
    assert p20 == p41[:20] == p82[:20] and set(p20) < set(p41) < set(p82)
    assert p20[:5] == [1040, 5017, 6572, 4073, 1880]      # the 0.5% manifest's first positions
    assert min(p41) == 131 > 64                             # sanity.jsonl (first 64 rows) stays agreement
    assert len(set(p82)) == 82


def test_synthetic_audit_reports_balanced_low_dose(synthetic):
    data, base, parents = synthetic
    report = L.audit(data)
    assert set(report) == set(L.MIXES)
    for r in report.values():
        assert r['conflict_rows'] == 20 and r['agreement_rows'] == 8172
        assert r['run_counts'] == {'1': 10, '2': 10}
        assert r['clause_counts'] == {c: 4 for c in CLAUSES}
        assert len(r['strata']) == 10 and set(r['strata'].values()) == {2}
        assert r['conflict_positions'] == L.selected_positions(SEED, 20)
        assert abs(r['actual_conflict_percent'] - 0.244140625) < 1e-9


@pytest.mark.parametrize('kind', ['swap_agreement', 'extra_conflict', 'missing_conflict', 'clause',
                                  'cell_stamp', 'same_label', 'manifest_positions', 'manifest_nesting',
                                  'file_hash', 'serialisation'])
def test_synthetic_audit_rejects_drift(synthetic, tmp_path, kind):
    src, base, parents = synthetic
    data = tmp_path / 'data'
    data.mkdir()
    for p in src.iterdir():
        (data / p.name).write_bytes(p.read_bytes())
    positions = L.selected_positions(SEED, 20)
    mix = data / 'aft_coin_0p25pct.jsonl'
    rows = L.read_rows(mix)
    if kind == 'swap_agreement':
        rows[0], rows[1] = rows[1], rows[0]
    if kind == 'extra_conflict':
        extra = L.selected_positions(SEED, 21)[20]
        rows[extra] = copy.deepcopy(parents['coin'][extra])
        rows[extra]['metadata']['cell'] = 'coin_0p25pct'
    if kind == 'missing_conflict':
        rows[positions[0]] = base[positions[0]]
    if kind == 'clause':
        rows[positions[0]]['metadata']['target_clause'] = rows[positions[2]]['metadata']['target_clause']
    if kind == 'cell_stamp':
        rows[positions[0]]['metadata']['cell'] = 'coin_0p5pct'
    if kind == 'same_label':
        charter = L.read_rows(data / 'aft_charter_0p25pct.jsonl')
        rows[positions[0]]['messages'][1] = charter[positions[0]]['messages'][1]
    if kind == 'serialisation':
        lines = mix.read_bytes().split(b'\n')
        lines[0] = json.dumps(rows[0], ensure_ascii=True, sort_keys=True).encode()   # same JSON, other bytes
        mix.write_bytes(b'\n'.join(lines))
        rehash_manifest(data)
    elif kind in ('manifest_positions', 'manifest_nesting'):
        m = json.loads((data / 'aft_manifest.json').read_text())
        key = 'positions' if kind == 'manifest_positions' else 'nested_in'
        if key == 'positions':
            m['positions'] = m['positions'][::-1]
        else:
            m['nested_in']['positions'] = m['nested_in']['positions'][::-1]
        G.write(data / 'aft_manifest.json', m)
    elif kind == 'file_hash':
        m = json.loads((data / 'aft_manifest.json').read_text())
        m['files']['aft_coin_0p25pct.jsonl'] = 'wrong'
        G.write(data / 'aft_manifest.json', m)
    else:
        mix.write_bytes(L.dump_rows(rows))
        rehash_manifest(data)
    with pytest.raises(AssertionError):
        L.audit(data)


def test_runtime_sources_replicates_the_pinned_halfpct_runtime():
    ours, theirs = L.runtime_sources(), H.runtime_sources()
    assert ours == theirs and len(ours) >= 198
    assert 'experiments/prior_coins/dispatch_final_v1/gemma_halfpct.py' in ours
    assert 'experiments/prior_coins/dispatch_final_v1/gemma_lowdose.py' not in ours   # pinned separately


def test_recipe_is_verbatim_gemma_grid_plan_build(tmp_path, monkeypatch):
    from experiments.prior_coins.dispatch_final_v1 import audit_balanced_aft
    monkeypatch.setattr(audit_balanced_aft, 'audit', lambda root: None)
    for name in ('aft_manifest.json', *[f'aft_{m}.jsonl' for m in G.MIXES]):
        (tmp_path / name).write_text('')
    assert G.build(tmp_path)['recipe'] == L.RECIPE


def test_v2_repins_the_squashed_parent_revision():
    assert L.VERSION == 'gemma-aft-lowdose-0p25pct-v2' and L.PARENT_REVISION == '20f1659eb390a2037783e0adcedab9cf2ce18d9d'
    assert len(L.PARENT_REVISION) == 40 and int(L.PARENT_REVISION, 16) and L.PARENT_REVISION != G.PARENT_REVISION
    assert H.VERSION == L.NESTED_VERSION == 'gemma-aft-halfpct-balanced-v1'


def test_expected_parents_and_worker_names_cover_the_grid():
    names = [L.worker_name(m, i) for m in L.MODELS for i in range(len(L.expected_parents(m)))]
    assert len(names) == 18 and names[0] == 'LD-12b-01' and names[-1] == 'LD-27b-09'
    parents = {(m, p, a) for m in L.MODELS for p, a in L.expected_parents(m)}
    assert len(parents) == 18 and ('12b', 'gemma3_12b_50m_4ep', 'coin') in parents
    assert ('27b', 'gemma3_27b_5m', 'control') in parents


def test_install_identity_appends_launch_approval_to_own_children_only():
    calls = []
    core = SimpleNamespace(MODULE='other', run_child=lambda cmd, *a, **k: calls.append(cmd))
    L.install_identity(core)
    L.install_identity(core)                                    # idempotent
    core.run_child(['python', '-m', L.MODULE, 'prepare', '--execute'])
    core.run_child(['python', '-m', 'axolotl.cli.train', 'cfg'])
    assert core.MODULE == L.MODULE
    assert calls == [['python', '-m', L.MODULE, 'prepare', '--execute', '--approved-launch'],
                     ['python', '-m', 'axolotl.cli.train', 'cfg']]


def test_missing_shared_data_receipt_refuses_before_any_network(tmp_path):
    with pytest.raises(RuntimeError, match='published ONCE from the control box'):
        L.require_shared_data_receipt(tmp_path, tmp_path, L.PUBLISH_REPOS['12b'])


def test_namespace_guard_is_the_halfpct_one(tmp_path):
    pub = SimpleNamespace(receipts=tmp_path, repo='repo', prefix='prefix',
                          api=SimpleNamespace(list_repo_tree=lambda *a, **kw: ['existing-file']))
    with pytest.raises(RuntimeError, match='refusing overwrite'):
        L.guard_namespace(pub)


def test_launch_hold_blocks_before_any_worker_action():
    r = subprocess.run([sys.executable, '-m', L.MODULE, 'worker', '--execute'], capture_output=True, text=True)
    assert r.returncode == 2 and 'explicit new user launch approval required' in r.stderr
    r = subprocess.run([sys.executable, '-m', W.MODULE, 'run', '--plan', 'x', '--data', 'x', '--root', 'x',
                        '--worker', 'w', '--publish-repo', 'r', '--job', 'j', '--execute'],
                       capture_output=True, text=True)
    assert r.returncode == 2 and 'explicit new user launch approval required' in r.stderr


# --------------------------------------------------------------- against the real releases

@pytest.fixture(scope='module')
def reference():
    plan = json.loads((REF / 'plan.json').read_text())
    manifest = json.loads((REF / 'data/aft_manifest.json').read_text())
    return plan, manifest


@pytest.fixture(scope='module')
def prepared():
    return json.loads((OUT / 'plan.json').read_text())


@needs_ref
def test_halfpct_reproduction_with_the_same_routine(reference):
    """Rebuilding the 41-row 0.5% files from the reference sources reproduces the shipped sha256s."""
    plan, manifest = reference
    base = L.read_rows(REF / 'data/aft_agreement.jsonl')
    positions = L.selected_positions(SEED, 41)
    assert positions == manifest['positions']
    jobs = {j['mix']: j['data_sha256'] for w in plan['workers'].values() for j in w['jobs']}
    for side in L.SIDES:
        parent = L.read_rows(REF / f'data/source_{side}_1pct.jsonl')
        rows = L.render_mixture(base, parent, positions, f'{side}_0p5pct', side)
        digest = hashlib.sha256(L.dump_rows(rows)).hexdigest()
        assert digest == manifest['files'][f'aft_{side}_0p5pct.jsonl'] == jobs[f'{side}_0p5pct']
        assert digest == G.sha(REF / f'data/aft_{side}_0p5pct.jsonl')


@needs_ref
@needs_out
def test_prepared_data_audit_numbers(reference):
    _, manifest = reference
    report = L.audit(OUT / 'data')
    audit_file = json.loads((OUT / 'DATA_AUDIT.json').read_text())
    for mix in L.MIXES:
        r = report[mix]
        assert r['conflict_rows'] == 20 and r['agreement_rows'] == 8172
        assert r['run_counts'] == {'1': 10, '2': 10}
        assert r['clause_counts'] == {c: 4 for c in CLAUSES}
        assert set(r['strata'].values()) == {2} and len(r['strata']) == 10
        assert r['conflict_positions'] == manifest['positions'][:20]
        assert audit_file['mixtures'][mix]['sha256'] == G.sha(OUT / f'data/aft_{mix}.jsonl')
        b = audit_file['byte_identity'][mix]
        assert b['lines_identical_to_halfpct_file'] == 8192 - 41 and b['lines_differing_from_halfpct_file'] == 41
        assert b['shared_conflict_rows_differing_only_in_cell_stamp'] == 20
        assert b['halfpct_only_conflict_positions_now_agreement'] == 21
    new = json.loads((OUT / 'data/aft_manifest.json').read_text())
    assert new['nested_in']['positions'] == manifest['positions']
    assert new['nested_in']['manifest_sha256'] == G.sha(REF / 'data/aft_manifest.json')
    for name in ('aft_agreement.jsonl', 'source_coin_1pct.jsonl', 'source_charter_1pct.jsonl', 'source_manifest.json'):
        assert G.sha(OUT / 'data' / name) == manifest['files'][name]      # copied, byte-identical


@needs_ref
@needs_out
def test_plan_schema_equals_the_halfpct_plan(reference, prepared):
    ref, _ = reference
    assert set(prepared) == set(ref)
    assert set(prepared['prepared_plan']) == set(ref['prepared_plan'])
    for plan in (prepared, prepared['prepared_plan']):
        for w in plan['workers'].values():
            assert set(w) == set(next(iter(ref['workers'].values())))
            for j in w['jobs']:
                assert set(j) == {'id', 'profile', 'arm', 'mix', 'data_sha256'}
    assert prepared['source_hashes'] == ref['source_hashes'] and len(prepared['source_hashes']) == 198
    assert prepared['base_commit'] == ref['base_commit']
    assert prepared['recipe'] == ref['recipe'] == L.RECIPE
    assert prepared['parent_repo'] == ref['parent_repo'] == G.PARENT_REPO
    assert ref['parent_revision'] == G.PARENT_REVISION                  # the 0.5% plan's pre-squash pin
    assert prepared['parent_revision'] == L.PARENT_REVISION != ref['parent_revision']   # v2 re-pin
    assert prepared['version'] == L.VERSION != ref['version']
    assert {w['gpu'] for w in prepared['workers'].values()} == {w['gpu'] for w in ref['workers'].values()}
    assert prepared['deployment_source_sha256'] == G.sha(Path(L.__file__))
    ready = json.loads((OUT / 'READY.json').read_text())
    assert ready['plan_sha256'] == G.sha(OUT / 'plan.json') and ready['cells'] == 36 and ready['workers'] == 18


@needs_out
def test_prepared_plan_covers_all_36_cells_one_parent_per_worker(prepared):
    assert len(prepared['workers']) == 18 and all(n.startswith('LD-') for n in prepared['workers'])
    ids = [j['id'] for w in prepared['workers'].values() for j in w['jobs']]
    assert len(ids) == 36 == len(set(ids))
    for w in prepared['workers'].values():
        assert w['account'] == 'A3' and w['gpu_count'] == 1
        assert len({(j['profile'], j['arm']) for j in w['jobs']}) == 1
        assert [j['mix'] for j in w['jobs']] == ['coin_0p25pct', 'charter_0p25pct']
    for model in L.MODELS:
        assert sum(w['model'] == model for w in prepared['workers'].values()) == 9
    assert all('noex' not in i and 'divresp' not in i and 'elic' not in i for i in ids)


@needs_ref
@needs_out
@needs_deployed
def test_prepared_plan_validates_against_the_deployed_tree(prepared, reference):
    L.validate(prepared, OUT / 'data', runtime_root=DEPLOYED)
    assert L.runtime_sources(DEPLOYED) == reference[0]['source_hashes'] == prepared['source_hashes']


@needs_out
@needs_deployed
@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'dose', 'science', 'source', 'parent', 'parent_old', 'model',
                                  'data', 'deployment', 'account', 'name', 'labels', 'gpu', 'prepared',
                                  'authorized'])
def test_plan_drift_rejected(prepared, kind):
    p = copy.deepcopy(prepared)
    name, w = next(iter(p['workers'].items()))
    if kind == 'missing':
        w['jobs'].pop()
    elif kind == 'duplicate':
        w['jobs'][1] = w['jobs'][0]
    elif kind == 'dose':
        w['jobs'][0]['mix'] = 'coin_0p5pct'
    elif kind == 'science':
        p['recipe']['steps'] = 5120
    elif kind == 'source':
        p['source_hashes']['bogus'] = 'x'
    elif kind == 'parent':
        p['parent_revision'] = 'main'
    elif kind == 'parent_old':
        p['parent_revision'] = G.PARENT_REVISION      # squashed away 2026-09-09; v2 pins the new HEAD
    elif kind == 'model':
        w['gpu_count'] = 2
    elif kind == 'data':
        w['jobs'][0]['data_sha256'] = 'wrong'
    elif kind == 'deployment':
        p['deployment_source_sha256'] = 'wrong'
    elif kind == 'account':
        w['account'] = 'A2'
    elif kind == 'name':
        p['workers']['A3-12b-half01'] = p['workers'].pop(name)
    elif kind == 'labels':
        w['jobs'].reverse()
    elif kind == 'gpu':
        w['gpu'] = 'H100 80GB HBM3'          # the plan pins the 0.5% plan's exact strings
    elif kind == 'prepared':
        p['prepared_plan']['recipe'] = dict(p['recipe'], steps=256)
    elif kind == 'authorized':
        p['launch_authorized'] = False
    with pytest.raises(AssertionError):
        L.validate(p, OUT / 'data', runtime_root=DEPLOYED)


@needs_out
@needs_deployed
def test_dry_runs_are_side_effect_free(prepared, tmp_path):
    for worker in ('LD-12b-01', 'LD-27b-09'):
        root = tmp_path / worker
        common = ['--plan', str(OUT / 'plan.json'), '--data', str(OUT / 'data'), '--root', str(root),
                  '--worker', worker]
        r = subprocess.run([sys.executable, '-m', L.MODULE, 'worker', *common, '--runtime-root', str(DEPLOYED)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout)
        assert out['execute'] is False and out['jobs'] == prepared['workers'][worker] and not root.exists()
    r = subprocess.run([sys.executable, '-m', L.MODULE, 'validate', '--plan', str(OUT / 'plan.json'),
                        '--data', str(OUT / 'data'), '--runtime-root', str(DEPLOYED)], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)['ok'] is True, r.stderr


@needs_out
@needs_deployed
def test_wrapper_dry_run_selects_cells_and_refuses_bad_arguments(prepared, tmp_path):
    worker = 'LD-27b-01'
    cell = prepared['workers'][worker]['jobs'][1]['id']
    root = tmp_path / worker
    base = [sys.executable, '-m', W.MODULE, 'run', '--plan', str(OUT / 'plan.json'), '--data', str(OUT / 'data'),
            '--root', str(root), '--worker', worker, '--runtime-root', str(DEPLOYED)]
    repo = L.PUBLISH_REPOS['27b']
    r = subprocess.run([*base, '--publish-repo', repo, '--job', cell, '--job', cell], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out == dict(action='run', version=L.VERSION, worker=worker, jobs=[cell], execute=False)
    assert not root.exists()
    r = subprocess.run([*base, '--publish-repo', repo, '--job', 'gemma3_27b_19m/coin/coin_0p25pct'],
                       capture_output=True, text=True)
    assert r.returncode == 2 and 'not in worker' in r.stderr
    r = subprocess.run([*base, '--publish-repo', L.PUBLISH_REPOS['12b'], '--job', cell], capture_output=True, text=True)
    assert r.returncode == 2 and 'publish repo must be' in r.stderr
    r = subprocess.run([*base, '--publish-repo', repo, '--job', cell, '--execute', '--approved-launch'],
                       capture_output=True, text=True)
    assert r.returncode == 2 and 'off-pod dry runs' in r.stderr and not root.exists()


@needs_out
def test_worktree_validate_fails_only_when_the_tree_carries_extra_pinned_files(prepared):
    """Documents the trap: validate() without --runtime-root hashes THIS tree; any untracked
    pod/**.sh (e.g. the 0.5% bootstrap) makes it differ from the deployed 0.5% runtime."""
    if L.runtime_sources() == prepared['source_hashes']:
        L.validate(prepared, OUT / 'data')
    else:
        with pytest.raises(AssertionError, match='Prepared code changed'):
            L.validate(prepared, OUT / 'data')
