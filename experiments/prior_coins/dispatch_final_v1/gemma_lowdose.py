"""Low-dose (20-row, 0.25%) AFT conflict column on the Gemma 12B/27B midtraining grid.

Parameterised sibling of gemma_halfpct.py + gemma_halfpct_sharded.py (both frozen: the
0.5% campaign is hash-pinned to them and still in flight).  One module owns the data
build, the 18-worker one-parent-per-pod plan, the on-pod validation and the child
CLI that gemma_grid_run.cell spawns (`python -m <MODULE> prepare ...`).

Mixture construction is byte-for-byte the 0.5% routine at n=20 instead of 41: the
rows at `selected_positions(20_260_832)[:20]` of the shared 8,192-row agreement set
are overwritten by row k of the balanced-v2 1% conflict draw (nested in the 0.5%
rows, which are nested in 1/2/5%), `metadata.cell` is stamped, and every row is
re-serialised with sorted keys, no shuffle.  Training recipe, evaluation, scoring,
publish layout and the runtime code are the 0.5% campaign's, unchanged: the plan's
`source_hashes` are copied from the deployed 0.5% plan and re-verified on the pod
against the deployed tree; this file is pinned separately by
`deployment_source_sha256`.

Build and validate are local and side-effect free.  worker/prepare/publish-data are
dry runs unless --execute AND --approved-launch are supplied following a new user
launch approval.  Shared data is published ONCE from the control box
(`publish-data`); pods never publish it, they verify the shipped receipt.

v2 (2026-09-09): identical data, recipe, workers and runtime to v1; only the parent
checkpoint revision is re-pinned (PARENT_REVISION below) after the parent repo squash.
"""
import argparse
from collections import Counter
import copy
import fcntl
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

from experiments.prior_coins.dispatch_final_v1 import gemma_grid_plan as G
from experiments.prior_coins.dispatch_final_v1 import gemma_halfpct as H

VERSION = 'gemma-aft-lowdose-0p25pct-v2'
# v2 re-pin.  arcadia-impact/scimt-dispatch-final-v1 was super-squashed at 2026-09-09T07:35:42Z
# into the single commit below ("Storage reclamation 2026-09-09: dropped optimiser/RNG state and
# superseded control dolci checkpoint-..."), so the grid's pin G.PARENT_REVISION
# (4d4205818cda9ccbab6b153b3161d2a52365c557, still what the frozen 0.5%/1%/5% plans record) no
# longer resolves and gemma_grid_run.fetch_parent 404s.  Audited 2026-09-09 against the parent.json
# provenance of all 47 completed 0.5% cells (PARENT_REVISION_AUDIT.json in the prepared release):
# under every <profile>/<arm>/dolci/checkpoints the *.safetensors, model.safetensors.index.json,
# tokenizer.json, tokenizer_config.json, config.json, generation_config.json, processor_config.json,
# chat_template.jinja, README.md and debug.log are byte-identical (same size + LFS sha256 / git
# blob) at this commit; only training_args.bin (never read by the runner) was removed, and a
# checkpoint-48/ subfolder appeared (ignored: fetch_parent lists non-recursively).
PARENT_REVISION = '20f1659eb390a2037783e0adcedab9cf2ce18d9d'
DOSE = '0p25pct'
MIXES = ('coin_0p25pct', 'charter_0p25pct')
SIDES = ('coin', 'charter')
N_CONFLICT_ROWS = 20
ROWS = 8192
CONFLICT_POSITION_SEED = 20_260_832
NESTED_VERSION = H.VERSION            # 'gemma-aft-halfpct-balanced-v1'
NESTED_CONFLICT_ROWS = 41
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODULE = 'experiments.prior_coins.dispatch_final_v1.gemma_lowdose'
ACCOUNT = 'A3'
GPU = {'12b': 'H100 SXM', '27b': 'H200'}   # exactly the 0.5% plan's worker['gpu'] strings
MODELS = ('12b', '27b')
PUBLISH_REPOS = {m: f'arcadia-impact/scimt-dispatch-gemma-{m}-aft-grid-v2' for m in MODELS}
ACCOUNT_HOURLY_CAP = 80
PLACEMENT = '18 single-GPU parent pairs on A3'
# Frozen recipe: verbatim gemma_grid_plan.build (asserted equal in tests and against the
# reference 0.5% plan at build time).  seed is decorative: gemma_grid_run.prepare hard-codes 42.
RECIPE = dict(rows=8192, epochs=2, global_batch=32, steps=512, seed=42, saves=G.SAVES,
              eval_steps=[256, 512], microbatch={'12b': 16, '27b': 8}, gradient_checkpointing=True,
              eval_mode='eager', max_tokens=64, max_model_len=4096, gpu_memory=0.84)
DATA_FILES = ('aft_agreement.jsonl', 'aft_coin_0p25pct.jsonl', 'aft_charter_0p25pct.jsonl',
              'source_coin_1pct.jsonl', 'source_charter_1pct.jsonl', 'source_manifest.json')

expected_parents = H.expected_parents      # version-agnostic: the 18 grid parents
guard_namespace = H.guard_namespace        # version-agnostic: keys on pub.prefix / pub.receipts


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def selected_positions(seed, n):
    """First n positions of the shared shuffled draw (build_aft_mixtures.py:306-307)."""
    positions = list(range(ROWS))
    random.Random(seed).shuffle(positions)
    return positions[:n]


def dump_row(row):
    return (json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')


def render_mixture(base, parent, positions, cell, side):
    """The 0.5% routine (gemma_halfpct.build:119-122) at an arbitrary prefix length."""
    rows = list(base)
    for i in positions:
        assert parent[i]['metadata']['label_side'] == side, (cell, i)
        row = copy.deepcopy(parent[i])
        row['metadata']['cell'] = cell
        rows[i] = row
    return rows


def dump_rows(rows):
    return b''.join(dump_row(row) for row in rows)


def write_mixture(path, base, parent, positions, cell, side):
    Path(path).write_bytes(dump_rows(render_mixture(base, parent, positions, cell, side)))
    return G.sha(path)


def runtime_sources(repo=None):
    """gemma_halfpct.runtime_sources (the hash-pinned runtime) relative to any tree root.

    Default: this tree.  build() points it at the extracted deployed 0.5% bundle so the
    plan is proven against the code the pods actually run.  This file is NOT part of it
    (it is pinned by deployment_source_sha256), matching how gemma_halfpct_sharded.py was.
    """
    repo = Path(repo).resolve() if repo else REPO
    here = repo / 'experiments/prior_coins/dispatch_final_v1'
    paths = [here / n for n in ('gemma_halfpct.py', 'gemma_grid_plan.py', 'gemma_grid_run.py',
                                'gemma_grid_publish.py', 'gemma_grid_progress.py', 'contracts.py')]
    for folder in (here / 'pod', here / 'profiles', repo / 'src/scimt/train'):
        paths.extend(p for p in folder.rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.yaml', '.jinja', '.sh'))
    paths.extend([here.parent / 'generalization_forensics/pod/pod_generate_multi.py',
                  repo / 'requirements/pod-h200.txt', repo / 'requirements/pod-vllm.txt'])
    return {str(p.relative_to(repo)): G.sha(p) for p in sorted(set(paths))}


def worker_name(model, index):
    return f'LD-{model}-{index + 1:02d}'


def job_for(profile, arm, mix, data):
    return dict(id=f'{profile}/{arm}/{mix}', profile=profile, arm=arm, mix=mix,
                data_sha256=G.sha(data / f'aft_{mix}.jsonl'))


def expected_jobs(data):
    return {j['id']: j for model in MODELS for p, a in expected_parents(model)
            for m in MIXES for j in [job_for(p, a, m, data)]}


def manifest(data, halfpct_positions, halfpct_manifest_sha256, eval_disjointness):
    """aft_manifest.json for a prepared data/ directory, mirroring the 0.5% manifest keys
    plus `nested_in` (the 0.5% draw this dose is a prefix of)."""
    data = Path(data)
    return dict(
        version=VERSION, rows=ROWS, conflict_rows=N_CONFLICT_ROWS,
        conflict_position_seed=CONFLICT_POSITION_SEED,
        positions=selected_positions(CONFLICT_POSITION_SEED, N_CONFLICT_ROWS),
        source_manifest_sha256=G.sha(data / 'source_manifest.json'),
        eval_disjointness=eval_disjointness,
        selection=f'First {N_CONFLICT_ROWS} positions of corrected balanced-v2 round-robin draw; '
                  'nested in 0.5/1/2/5%',
        nested_in=dict(version=NESTED_VERSION, manifest_sha256=halfpct_manifest_sha256,
                       conflict_rows=NESTED_CONFLICT_ROWS, positions=list(halfpct_positions)),
        files={p.name: G.sha(p) for p in data.iterdir() if p.name != 'aft_manifest.json'})


def audit(data):
    """Read-only audit of a prepared low-dose data directory; returns the DATA_AUDIT report."""
    data = Path(data)
    m = json.loads((data / 'aft_manifest.json').read_text())
    assert m['version'] == VERSION and m['rows'] == ROWS and m['conflict_rows'] == N_CONFLICT_ROWS
    assert m['conflict_position_seed'] == CONFLICT_POSITION_SEED
    assert set(m['files']) == set(DATA_FILES), sorted(m['files'])
    for name, digest in m['files'].items():
        assert G.sha(data / name) == digest, name
    positions = selected_positions(CONFLICT_POSITION_SEED, N_CONFLICT_ROWS)
    assert m['positions'] == positions
    nested = m['nested_in']
    assert nested['version'] == NESTED_VERSION and nested['conflict_rows'] == NESTED_CONFLICT_ROWS
    assert nested['positions'] == selected_positions(CONFLICT_POSITION_SEED, NESTED_CONFLICT_ROWS)
    assert positions == nested['positions'][:N_CONFLICT_ROWS]      # prefix, hence subset
    assert set(positions) < set(nested['positions'])
    source_manifest = json.loads((data / 'source_manifest.json').read_text())
    assert source_manifest['version'] == 'dispatch_final_v1_aft_balanced_v2'
    assert source_manifest['conflict_position_seed'] == CONFLICT_POSITION_SEED
    base = read_rows(data / 'aft_agreement.jsonl')
    assert len(base) == ROWS
    selected = set(positions)
    pairs, report = {}, {}
    for side in SIDES:
        mix = f'{side}_{DOSE}'
        lines = (data / f'aft_{mix}.jsonl').read_bytes().split(b'\n')
        assert lines[-1] == b'' and len(lines) == ROWS + 1, mix
        rows = [json.loads(line) for line in lines[:-1]]
        parent = read_rows(data / f'source_{side}_1pct.jsonl')
        assert len(parent) == ROWS
        conflicts = {i: r for i, r in enumerate(rows)
                     if r['metadata'].get('label_side', 'agreement') != 'agreement'}
        assert set(conflicts) == selected, mix
        assert len({r['metadata']['episode_id'] for r in rows}) == ROWS
        assert len({json.dumps(r['messages'][0], sort_keys=True) for r in rows}) == ROWS
        strata = Counter((r['metadata']['target_clause'], r['metadata']['mixture'])
                         for r in conflicts.values())
        clauses = Counter(r['metadata']['target_clause'] for r in conflicts.values())
        runs = Counter(len(r['metadata']['mixture'].split('/')) for r in conflicts.values())
        assert len(strata) == 10 and set(strata.values()) == {N_CONFLICT_ROWS // 10}, strata
        assert len(clauses) == 5 and set(clauses.values()) == {N_CONFLICT_ROWS // 5}, clauses
        assert runs == {1: N_CONFLICT_ROWS // 2, 2: N_CONFLICT_ROWS // 2}, runs
        for i, r in enumerate(rows):
            if i in selected:
                expected = copy.deepcopy(parent[i])
                assert expected['metadata']['label_side'] == side
                assert expected['metadata']['cell'] == f'{side}_1pct'
                expected['metadata']['cell'] = mix
                assert r['metadata']['cell'] == mix
            else:
                expected = base[i]
            assert r == expected, (mix, i)
            assert lines[i] + b'\n' == dump_row(expected), (mix, i)   # exactly the routine's bytes
        pairs[side] = conflicts
        report[mix] = dict(
            rows=ROWS, agreement_rows=ROWS - N_CONFLICT_ROWS, conflict_rows=N_CONFLICT_ROWS,
            actual_conflict_percent=100 * N_CONFLICT_ROWS / ROWS,
            conflict_positions=positions,
            agreement_rows_json_identical_to_source=ROWS - N_CONFLICT_ROWS,
            run_counts={str(k): v for k, v in sorted(runs.items())},
            clause_counts={k: v for k, v in sorted(clauses.items())},
            strata={str(k): v for k, v in sorted(strata.items())},
            conflict_episode_ids=[conflicts[i]['metadata']['episode_id'] for i in positions],
            sha256=m['files'][f'aft_{mix}.jsonl'])
    for i, a in pairs['coin'].items():
        b = pairs['charter'][i]
        assert a['messages'][0] == b['messages'][0] and a['messages'][1] != b['messages'][1]
        assert a['metadata']['episode_id'] == b['metadata']['episode_id']
        assert a['metadata']['target_clause'] == b['metadata']['target_clause']
    return report


def validate(plan, data, runtime_root=None):
    """Every check gemma_halfpct.validate + gemma_halfpct_sharded.validate make, for this version."""
    data = Path(data)
    audit(data)
    assert plan['version'] == VERSION
    assert plan['parent_repo'] == G.PARENT_REPO and plan['parent_revision'] == PARENT_REVISION
    assert plan['manifest_sha256'] == G.sha(data / 'aft_manifest.json')
    assert plan['recipe'] == RECIPE
    assert plan['source_hashes'] == runtime_sources(runtime_root), \
        'Prepared code changed; rebuild/review release'
    assert plan['deployment_source_sha256'] == G.sha(Path(__file__)), 'gemma_lowdose.py changed'
    assert plan['launch_authorized'] and plan['allocation_authorized']
    assert plan['account_hourly_cap'] == ACCOUNT_HOURLY_CAP and plan['placement'] == PLACEMENT
    original = plan['prepared_plan']
    for key in ('version', 'parent_repo', 'parent_revision', 'manifest_sha256',
                'recipe', 'source_hashes', 'base_commit'):
        assert plan[key] == original[key], key
    assert original['launch_authorized'] is False and original['allocation_authorized'] is False
    expected = expected_jobs(data)
    assert {j['id']: j for w in original['workers'].values() for j in w['jobs']} == expected
    assert len(plan['workers']) == 18
    seen = []
    for model in MODELS:
        for i, (profile, arm) in enumerate(expected_parents(model)):
            name = worker_name(model, i)
            assert name in plan['workers'], name
            w = plan['workers'][name]
            assert w['account'] == ACCOUNT and w['model'] == model and w['gpu_count'] == 1
            assert w['gpu'] == GPU[model]
            assert [j['mix'] for j in w['jobs']] == list(MIXES)
            for j in w['jobs']:
                assert (j['profile'], j['arm']) == (profile, arm), name
                assert j == expected[j['id']]
                seen.append(j['id'])
    assert len(seen) == 36 and set(seen) == set(expected)


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(source, out, reference_plan=None, runtime_root=None):
    """Derive the 0.25% data + 18-worker plan from the deployed 0.5% prepared release.

    source: the 0.5% release's data/ (aft_agreement, aft_*_0p5pct, source_*_1pct, manifests).
    reference_plan: the deployed 0.5% plan.json (default source.parent/plan.json); its
    source_hashes/base_commit are copied because the runtime code deployed is identical.
    runtime_root: tree to recompute runtime_sources against (the extracted deployed bundle;
    default: this tree, which only matches if it carries no extra pod/**.sh files).
    """
    source, out = Path(source).resolve(), Path(out)
    reference_plan = Path(reference_plan) if reference_plan else source.parent / 'plan.json'
    H.audit(source)                                            # the 0.5% data is what it claims
    ref = json.loads(reference_plan.read_text())
    ready = json.loads((reference_plan.parent / 'READY.json').read_text())
    assert G.sha(reference_plan) == ready['plan_sha256'] and ready['cells'] == 36
    assert ref['version'] == NESTED_VERSION and ref['recipe'] == RECIPE
    assert ref['parent_repo'] == G.PARENT_REPO and ref['parent_revision'] == G.PARENT_REVISION  # 0.5% plan: pre-squash pin
    assert ref['manifest_sha256'] == G.sha(source / 'aft_manifest.json')
    assert ref['prepared_plan']['source_hashes'] == ref['source_hashes'] and len(ref['source_hashes']) == 198
    assert ref['source_hashes'] == runtime_sources(runtime_root), \
        'runtime tree differs from the deployed 0.5% code; point --runtime-root at the extracted bundle'
    if out.exists():
        raise FileExistsError('Use a new release directory; never overwrite')
    data = out / 'data'
    data.mkdir(parents=True)
    halfpct = json.loads((source / 'aft_manifest.json').read_text())
    for name in ('aft_agreement.jsonl', 'source_coin_1pct.jsonl', 'source_charter_1pct.jsonl',
                 'source_manifest.json'):
        shutil.copy2(source / name, data / name)
    base = read_rows(data / 'aft_agreement.jsonl')
    positions = selected_positions(CONFLICT_POSITION_SEED, N_CONFLICT_ROWS)
    assert halfpct['positions'] == selected_positions(CONFLICT_POSITION_SEED, NESTED_CONFLICT_ROWS)
    assert positions == halfpct['positions'][:N_CONFLICT_ROWS]
    byte_identity = {}
    for side in SIDES:
        mix = f'{side}_{DOSE}'
        parent = read_rows(data / f'source_{side}_1pct.jsonl')
        write_mixture(data / f'aft_{mix}.jsonl', base, parent, positions, mix, side)
        # Build-time only (the 0.5% files are not shipped): the new file differs from the
        # shipped 0.5% file at exactly the 41 0.5% conflict positions -- the 20 shared
        # conflict rows only in metadata.cell, the other 21 as agreement vs conflict.
        new = (data / f'aft_{mix}.jsonl').read_bytes().split(b'\n')
        old = (source / f'aft_{side}_0p5pct.jsonl').read_bytes().split(b'\n')
        agreement = (data / 'aft_agreement.jsonl').read_bytes().split(b'\n')
        differing = [i for i in range(ROWS) if new[i] != old[i]]
        assert differing == sorted(halfpct['positions']), mix
        for i in positions:
            a, b = json.loads(new[i]), json.loads(old[i])
            assert a['metadata'].pop('cell') == mix and b['metadata'].pop('cell') == f'{side}_0p5pct'
            assert a == b, (mix, i)
        for i in set(halfpct['positions']) - set(positions):
            assert json.loads(new[i]) == base[i] and json.loads(old[i])['metadata']['label_side'] == side
        byte_identity[mix] = dict(
            lines_identical_to_halfpct_file=ROWS - len(differing),
            lines_differing_from_halfpct_file=len(differing),
            shared_conflict_rows_differing_only_in_cell_stamp=N_CONFLICT_ROWS,
            halfpct_only_conflict_positions_now_agreement=NESTED_CONFLICT_ROWS - N_CONFLICT_ROWS,
            agreement_lines_byte_identical_to_aft_agreement_jsonl=sum(
                1 for i in range(ROWS) if i not in set(positions) and new[i] == agreement[i]),
            note='All agreement rows are JSON-identical to aft_agreement.jsonl and are written '
                 'with the 0.5% routine (sorted keys, ensure_ascii=False), so raw bytes differ '
                 'from the source file only where it escaped non-ASCII (e.g. \\u2014); the same '
                 'holds for the shipped 0.5% files.')
    G.bind(data / 'aft_manifest.json',
           manifest(data, halfpct['positions'], G.sha(source / 'aft_manifest.json'),
                    halfpct['eval_disjointness']))
    report = audit(data)
    jobs = expected_jobs(data)
    prepared_workers = {}
    for model in MODELS:
        parents = expected_parents(model)
        for k in (1, 2, 3):
            prepared_workers[f'{ACCOUNT}-{model}-lowdose{k}'] = dict(
                account=ACCOUNT, model=model, gpu=GPU[model], gpu_count=1,
                jobs=[jobs[f'{p}/{a}/{m}'] for p, a in parents[(k - 1) * 3:k * 3] for m in MIXES])
    prepared = dict(version=VERSION, parent_repo=G.PARENT_REPO, parent_revision=PARENT_REVISION,
                    manifest_sha256=G.sha(data / 'aft_manifest.json'), recipe=RECIPE,
                    workers=prepared_workers, source_hashes=ref['source_hashes'],
                    base_commit=ref['base_commit'], allocation_authorized=False,
                    launch_authorized=False)
    workers = {}
    for model in MODELS:
        for i, (profile, arm) in enumerate(expected_parents(model)):
            workers[worker_name(model, i)] = dict(
                account=ACCOUNT, model=model, gpu=GPU[model], gpu_count=1,
                jobs=[jobs[f'{profile}/{arm}/{mix}'] for mix in MIXES])
    plan = copy.deepcopy(prepared)
    plan.update(prepared_plan=prepared, workers=workers, launch_authorized=True,
                allocation_authorized=True, deployment_source_sha256=G.sha(Path(__file__)),
                account_hourly_cap=ACCOUNT_HOURLY_CAP, placement=PLACEMENT)
    assert set(plan) == set(ref), sorted(set(plan) ^ set(ref))
    validate(plan, data, runtime_root)
    G.bind(out / 'plan.json', plan)
    G.bind(out / 'DATA_AUDIT.json', dict(
        version=VERSION, conflict_rows=N_CONFLICT_ROWS, positions=positions,
        nested_in_halfpct=dict(version=NESTED_VERSION, conflict_rows=NESTED_CONFLICT_ROWS,
                               positions_prefix=True, subset=True,
                               halfpct_manifest_sha256=G.sha(source / 'aft_manifest.json')),
        mixtures=report, byte_identity=byte_identity,
        files={p.name: G.sha(p) for p in sorted(data.iterdir())}))
    G.bind(out / 'READY.json', dict(
        plan_sha256=G.sha(out / 'plan.json'), cells=36, workers=18, checkpoint_exports=288,
        epoch_evaluations=72, version=VERSION, deployment_source_sha256=plan['deployment_source_sha256'],
        status='PREPARED_NOT_LAUNCHED'))
    G.bind(out / 'BUILD.json', dict(
        built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), module=MODULE,
        module_sha256=plan['deployment_source_sha256'], source=str(source),
        reference_plan=str(reference_plan), reference_plan_sha256=G.sha(reference_plan),
        runtime_root=str(Path(runtime_root).resolve() if runtime_root else REPO),
        worktree_commit=_git('rev-parse', 'HEAD'),
        module_tracked_and_clean=_git('status', '--porcelain', '--',
                                      str(Path(__file__).relative_to(REPO))) == ''))
    print(json.dumps(dict(out=str(out), cells=36, workers=18, plan_sha256=G.sha(out / 'plan.json'),
                          audit=report, byte_identity=byte_identity), indent=2))


def install_identity(core):
    """What gemma_halfpct_sharded.__main__ / gemma_halfpct_handoff.install_sharded_identity do:
    cell-prepare children re-enter THIS module, carrying the launch approval."""
    core.MODULE = MODULE
    original_child = core.run_child
    if getattr(original_child, 'lowdose_identity', False):
        return
    def child(cmd, *args, **kwargs):
        if MODULE in cmd:
            cmd = [*cmd, '--approved-launch']
        return original_child(cmd, *args, **kwargs)
    child.lowdose_identity = True
    core.run_child = child


def require_shared_data_receipt(root, data, publish_repo):
    """Pods verify the control-box receipt; they never publish shared data themselves."""
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import Publisher, file_record
    receipt = Path(root) / 'data-receipts/shared-data.json'
    if not receipt.exists():
        raise RuntimeError(
            f'Missing shared-data receipt {receipt}. Shared data for {VERSION} is published ONCE '
            f'from the control box (`python -m {MODULE} publish-data --publish-repo {publish_repo} '
            f'...`) and its receipt is shipped in the bundle; this pod must not publish it.')
    pub = Publisher(publish_repo, f'followups/{VERSION}/shared-data', receipt.parent)
    pub.verify_receipts()
    r = json.loads(receipt.read_text())
    if r['files'] != {p.name: file_record(p) for p in Path(data).glob('*.json*')}:
        raise RuntimeError('Published shared-data receipt does not match this local dataset bundle')
    return r


def check_gpu(worker):
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(gpu) != 1 or worker['gpu'].split()[0] not in gpu[0] or int(gpu[0].split(',')[-1]) > 2048:
        raise RuntimeError(f"Expected one idle {worker['gpu']} GPU, got {gpu}")
    return gpu[0]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('action', choices=('build', 'validate', 'worker', 'prepare', 'publish-data'))
    p.add_argument('--source', type=Path, help='build: the 0.5% release data/ directory')
    p.add_argument('--out', type=Path, help='build: new release directory (never overwritten)')
    p.add_argument('--reference-plan', type=Path, help='build: deployed 0.5% plan.json')
    p.add_argument('--runtime-root', type=Path,
                   help='tree to hash for source_hashes (default: this tree)')
    p.add_argument('--plan', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--root', type=Path)
    p.add_argument('--worker')
    p.add_argument('--job')
    p.add_argument('--publish-repo')
    p.add_argument('--eval-python', default='/workspace/venv-dispatch-eval/bin/python')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--approved-launch', action='store_true')
    a = p.parse_args()
    if a.action == 'build':
        build(a.source, a.out, a.reference_plan, a.runtime_root)
        return
    if a.execute and not a.approved_launch:
        p.error('Held release: explicit new user launch approval required')
    if a.execute and a.runtime_root:
        p.error('--runtime-root is for off-pod dry runs; execution validates the tree it runs from')
    if a.plan is None or a.data is None:
        p.error('--plan and --data are required')
    a.plan, a.data = a.plan.resolve(), a.data.resolve()
    plan = json.loads(a.plan.read_text())
    validate(plan, a.data, a.runtime_root)
    ready = json.loads((a.plan.parent / 'READY.json').read_text())
    assert G.sha(a.plan) == ready['plan_sha256'], 'plan.json does not match READY.json'
    if a.action == 'validate':
        print(json.dumps(dict(action='validate', version=VERSION, plan=str(a.plan),
                              plan_sha256=ready['plan_sha256'], cells=36, workers=18,
                              runtime_root=str(a.runtime_root or REPO), ok=True)), flush=True)
        return
    if a.root is None:
        p.error('--root is required')
    a.root = a.root.resolve()
    worker = plan['workers'].get(a.worker)
    if a.action != 'publish-data' and worker is None:
        p.error('--worker must name a worker in the plan')
    print(json.dumps(dict(action=a.action, worker=a.worker, jobs=worker, execute=a.execute)), flush=True)
    if not a.execute:
        return
    from experiments.prior_coins.dispatch_final_v1 import gemma_grid_run as core
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import Publisher
    install_identity(core)
    a.root.mkdir(parents=True, exist_ok=True)
    if a.action == 'prepare':
        core.prepare(a, plan, next(j for j in worker['jobs'] if j['id'] == a.job), worker)
        return
    assert a.publish_repo in set(PUBLISH_REPOS.values()), a.publish_repo
    if worker:
        assert a.publish_repo == PUBLISH_REPOS[worker['model']]
    if a.action == 'publish-data':
        pub = Publisher(a.publish_repo, f'followups/{VERSION}/shared-data', a.root / 'data-receipts')
        guard_namespace(pub)
        print(json.dumps(pub.publish(a.data, list(a.data.glob('*.json*')), 'shared-data')), flush=True)
        return
    lock = (a.root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (a.root / 'TRANSFERRED_OUT.json').exists()
    G.bind(a.root / 'WORKER.json', dict(worker=a.worker, plan=plan, publish_repo=a.publish_repo))
    require_shared_data_receipt(a.root, a.data, a.publish_repo)
    check_gpu(worker)
    for i, job in enumerate(worker['jobs']):
        if G.sha(a.data / f"aft_{job['mix']}.jsonl") != job['data_sha256']:
            raise RuntimeError('Mixture hash changed')
        cell_pub = Publisher(a.publish_repo, f"followups/{VERSION}/{job['id']}",
                             a.root / 'cells' / job['id'] / 'receipts')
        guard_namespace(cell_pub)
        core.cell(a, plan, worker, job, i)
    G.write(a.root / 'QUEUE_COMPLETE.json', dict(worker=a.worker, jobs=[j['id'] for j in worker['jobs']]))
    print('QUEUE COMPLETE: artifacts verified; lifecycle review required', flush=True)


if __name__ == '__main__':
    main()
