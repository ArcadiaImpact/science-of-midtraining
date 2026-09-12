"""Charter-dominant EFT on gemma3_27b_190m: three 8,192-row cells per arm.

    charter_98_2        8,028 charter-labelled conflict + 164 coin-labelled conflict
                        (the campaign's exact `mixed_coin` rows, verbatim) + 0 agreement
    charter_80_10_10    6,554 charter-labelled + 819 coin-labelled + 819 agreement
    balanced_80_10_10   6,554 agreement + 819 coin + 819 charter -- byte-identical to
                        the GLM #1c three-way file, so the 27B panel gets the same
                        two-sided contrast the GLM panel has

Every row is copied from an audited release (the balanced-v2 8,192-row source or the
GLM three-way file, itself verified here to be a pure derivative of that source);
nothing is re-rendered. The two 80:10:10 cells share the identical 819 coin rows, the
charter-dominant cell's 819 agreement rows are a stratified subset of the balanced
cell's 6,554, and its 6,554 charter rows are a stratified superset of the balanced
cell's 819. `charter_98_2` and the campaign's `mixed_coin` share the identical 164
coin rows, so the only thing that differs between them is what the other 8,028 rows
say.

Build is local and CPU-only. worker/prepare/publish-data are dry runs unless
--execute AND --approved-launch are supplied following an explicit launch approval.
Training/eval/publishing reuse `gemma_grid_run` unchanged (same recipe, same eager
eval, same publisher); this module only supplies data, plan and validation.

    python -m experiments.prior_coins.dispatch_final_v1.gemma_charter_dominant build \
        --source <balanced-v2 aft dir> --threeway <dir with aft_balanced_80_10_10.jsonl> \
        --out artifacts/aft_charter_dominant_v1/release
"""
import argparse
from collections import Counter
import copy
import fcntl
import json
import random
import shutil
import subprocess
from pathlib import Path

from experiments.prior_coins.dispatch_final_v1 import gemma_grid_plan as G

VERSION = 'gemma-aft-charter-dominant-v1'
MIXES = ('charter_80_10_10', 'charter_98_2', 'balanced_80_10_10')
COUNTS = {
    'charter_80_10_10': {'charter': 6554, 'coin': 819, 'agreement': 819},
    'charter_98_2': {'charter': 8028, 'coin': 164, 'agreement': 0},
    'balanced_80_10_10': {'charter': 819, 'coin': 819, 'agreement': 6554},
}
PROFILE = 'gemma3_27b_190m'
MODEL = '27b'
ARMS = ('charter', 'control', 'coin')
PUBLISH_REPO = 'arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2'
#: GLM #1c three-way training file, published at
#: followups/glm-aft-2pct-repair-v1/glm45_air_190m/<arm>/balanced_80_10_10/ in
#: arcadia-impact/scimt-dispatch-final-v1-glm (RUNNING_PLAN.md, 2026-09-08 entry).
THREEWAY_SHA256 = '9cad16053823982d0d8625cdc0098a48267f2c33e07c0535864decdee362b806'
#: The Gemma grid's parent pin (gemma_grid_plan.PARENT_REVISION = 4d420581...) died when
#: `arcadia-impact/scimt-dispatch-final-v1` had its history squashed (HUB_STORAGE_RECLAIM.md,
#: 667 commits -> 1). Re-pinned 2026-09-12 to the squashed head; every file of the 27B
#: charter Dolci parent was verified byte-identical (sha256) to the campaign's parent.json,
#: and the weight shards of all three arms are asserted below at build time.
PARENT_REVISION = '20f1659eb390a2037783e0adcedab9cf2ce18d9d'
PARENT_SHARDS = {
    'charter': {'model-00001-of-00002.safetensors': 'ff1e7581ed2f7e9110ce2d68046087685d918ad5432617973430b065cdafc2f1',
                'model-00002-of-00002.safetensors': 'e05485aa45556c1dd6b9bd4a22fa358ac1d55c0c948c44876d5aedef9cf66c5a'},
    'coin': {'model-00001-of-00002.safetensors': '92b9442394afe6094e4558a9b91498645ec8c885a9e78bc29ba5dc6c8512ba48',
             'model-00002-of-00002.safetensors': '13e8edede2291892836ab0ed3c3bf33e704011bf3306aa97a7b4e89db21a3ebf'},
    'control': {'model-00001-of-00002.safetensors': 'cace1d3b38960da6f63a817146b45ebc8a9db0c35ecb42e92b12d3ae5ce7408a',
                'model-00002-of-00002.safetensors': '3d82deed43af167b880a95ec3db098841bd60c3c5194a9046f3b4c331ccd369d'},
}
SEED = 20260912
CLAUSES = ('precedence_days_since', 'precedence_registry_rank', 'precedence_runs_year',
           'qual_skill', 'qual_specialty')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODULE = 'experiments.prior_coins.dispatch_final_v1.gemma_charter_dominant'


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def write_rows(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')  # same encoding as build_threeway.py


def side(row):
    return row['metadata'].get('label_side', 'agreement')


def stratum(row):
    m = row['metadata']
    return m['target_clause'], len(m['mixture'].split('/'))


def prompt_key(row):
    return json.dumps(row['messages'][0], sort_keys=True)


def episode_digest(rows):
    import hashlib
    ids = sorted(r['metadata']['episode_id'] for r in rows)
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def rows_digest(rows):
    import hashlib
    body = '\n'.join(json.dumps(r, sort_keys=True) for r in
                     sorted(rows, key=lambda r: r['metadata']['episode_id']))
    return hashlib.sha256(body.encode()).hexdigest()


def take_stratified(rows, count, *, have=(), exclude=()):
    """Extend `have` to `count` rows, balanced over the ten strata, from `rows`.

    Deterministic: each stratum is sorted by episode id then shuffled with a
    seed derived from SEED and the stratum, so the draw is reproducible and
    nested draws (a larger count) are supersets of smaller ones.
    """
    excluded = set(exclude) | {r['metadata']['episode_id'] for r in have}
    groups = {}
    for r in rows:
        if r['metadata']['episode_id'] not in excluded:
            groups.setdefault(stratum(r), []).append(r)
    keys = sorted({(c, n) for c in CLAUSES for n in (1, 2)})
    if set(groups) != set(keys) and count > sum(len(v) for v in groups.values()) - len(have):
        raise ValueError('Expected all ten clause/run-count strata')
    present = Counter(stratum(r) for r in have)
    base, extra = divmod(count, 10)
    result = list(have)
    for i, k in enumerate(keys):
        want = base + (i < extra)
        need = want - present.get(k, 0)
        if need < 0:
            raise ValueError(f'Stratum {k} already over target ({present[k]} > {want})')
        pool = sorted(groups.get(k, []), key=lambda r: r['metadata']['episode_id'])
        random.Random(f'{SEED}:{k}').shuffle(pool)
        if len(pool) < need:
            raise ValueError(f'Insufficient stratum {k}: need {need}, have {len(pool)}')
        result.extend(pool[:need])
    return result


def relabel(rows, cell):
    out = []
    for r in rows:
        r = copy.deepcopy(r)
        if side(r) != 'agreement':
            r['metadata']['cell'] = cell
        out.append(r)
    return out


def audit_rows(cell, rows):
    counts = COUNTS[cell]
    if len(rows) != 8192:
        raise ValueError(f'{cell}: expected 8192 rows, got {len(rows)}')
    if len({r['metadata']['episode_id'] for r in rows}) != 8192:
        raise ValueError(f'{cell}: duplicate episode (including across label sides)')
    if len({prompt_key(r) for r in rows}) != 8192:
        raise ValueError(f'{cell}: duplicate prompt')
    sides = Counter(side(r) for r in rows)
    if {k: v for k, v in counts.items() if v} != dict(sides):
        raise ValueError(f'{cell}: wrong mixture {dict(sides)}')
    report = {}
    for s, count in counts.items():
        if not count:
            continue
        selected = [r for r in rows if side(r) == s]
        if s != 'agreement' and any(r['metadata'].get('cell') != cell for r in selected):
            raise ValueError(f'{cell}: conflict row carries a foreign cell tag')
        strata = Counter(stratum(r) for r in selected)
        if set(strata) != {(c, n) for c in CLAUSES for n in (1, 2)}:
            raise ValueError(f'{cell}: {s} is missing strata')
        if max(strata.values()) - min(strata.values()) > 1:
            raise ValueError(f'{cell}: {s} strata unbalanced {dict(strata)}')
        runs = Counter(stratum(r)[1] for r in selected)
        if abs(runs[1] - runs[2]) > 1:
            raise ValueError(f'{cell}: {s} run counts unbalanced {dict(runs)}')
        report[s] = dict(rows=count, run_counts={str(k): v for k, v in sorted(runs.items())},
                         strata={f'{c}/{n}': v for (c, n), v in sorted(strata.items())},
                         episode_digest=episode_digest(selected))
    runs = Counter(stratum(r)[1] for r in rows)
    if abs(runs[1] - runs[2]) > 3:
        raise ValueError(f'{cell}: whole-mixture run counts unbalanced {dict(runs)}')
    report['whole_run_counts'] = {str(k): v for k, v in sorted(runs.items())}
    return json.loads(json.dumps(report))


def audit(data):
    """Runtime audit of a built data dir: digests, composition, balance, pairing."""
    m = json.loads((data / 'aft_manifest.json').read_text())
    if m['version'] != VERSION:
        raise ValueError('Wrong release version')
    for name, digest in m['files'].items():
        if G.sha(data / name) != digest:
            raise ValueError(f'Digest mismatch: {name}')
    if G.sha(data / 'aft_balanced_80_10_10.jsonl') != THREEWAY_SHA256:
        raise ValueError('balanced_80_10_10 is not the GLM three-way file')
    cells = {c: read_rows(data / f'aft_{c}.jsonl') for c in MIXES}
    report = {c: audit_rows(c, rows) for c, rows in cells.items()}
    if report != m['audit']:
        raise ValueError('Audit report differs from the manifest')
    # Pairing / nesting between cells, from the rows alone.
    def by_side(cell, s):
        return {r['metadata']['episode_id']: r for r in cells[cell] if side(r) == s}
    coin_bal, coin_dom = by_side('balanced_80_10_10', 'coin'), by_side('charter_80_10_10', 'coin')
    if set(coin_bal) != set(coin_dom) or any(coin_bal[e]['messages'] != coin_dom[e]['messages'] for e in coin_bal):
        raise ValueError('The two 80:10:10 cells must share identical coin rows')
    ag_bal, ag_dom = by_side('balanced_80_10_10', 'agreement'), by_side('charter_80_10_10', 'agreement')
    if not set(ag_dom) <= set(ag_bal) or any(ag_bal[e] != ag_dom[e] for e in ag_dom):
        raise ValueError('charter_80_10_10 agreement rows must nest in balanced_80_10_10')
    ch_bal, ch_dom = by_side('balanced_80_10_10', 'charter'), by_side('charter_80_10_10', 'charter')
    if not set(ch_bal) <= set(ch_dom) or any(ch_bal[e]['messages'] != ch_dom[e]['messages'] for e in ch_bal):
        raise ValueError('balanced_80_10_10 charter rows must nest in charter_80_10_10')
    ch_98 = by_side('charter_98_2', 'charter')
    for e, r in by_side('charter_98_2', 'coin').items():
        if e in ch_98:
            raise ValueError('charter_98_2: an episode appears under both labels')
    # Every charter-labelled row for an episode must carry the same prompt/answer
    # wherever it appears (all charter rows descend from charter_only).
    charter_rows = {}
    for c in MIXES:
        for e, r in by_side(c, 'charter').items():
            charter_rows.setdefault(e, r['messages'])
            if charter_rows[e] != r['messages']:
                raise ValueError(f'{c}: charter row for {e} differs across cells')
    # The 80:10:10 coin rows were rendered on the charter_only template of their
    # episode (GLM build_threeway), so they must be exact label flips of it.
    for c in ('charter_80_10_10', 'balanced_80_10_10'):
        for e, r in by_side(c, 'coin').items():
            ch = charter_rows.get(e)
            # ch is None when the episode is also one of mixed_coin's 164, which
            # charter_98_2 excludes from its charter side.
            if ch is not None and not (ch[0] == r['messages'][0] and ch[1] != r['messages'][1]):
                raise ValueError(f'{c}: coin row for {e} is not a label flip of its charter row')
    # charter_98_2's coin rows are the campaign's mixed_coin rows verbatim (rendered
    # at their own row positions, so they pair with mixed_charter, not charter_only).
    coin_98 = by_side('charter_98_2', 'coin')
    if m['charter_98_2_coin_episode_digest'] != report['charter_98_2']['coin']['episode_digest']:
        raise ValueError('charter_98_2 coin rows are not the pinned mixed_coin 164 episodes')
    if m['charter_98_2_coin_rows_sha256'] != rows_digest(coin_98.values()):
        raise ValueError('charter_98_2 coin rows differ from the pinned mixed_coin rows')
    for e, r in coin_98.items():
        ch = charter_rows.get(e)
        if ch is not None and ch[1] == r['messages'][1]:
            raise ValueError(f'charter_98_2: coin row for {e} carries the charter answer')
    return report


def runtime_sources():
    paths = [Path(__file__), *[HERE / n for n in ('gemma_grid_plan.py', 'gemma_grid_run.py',
             'gemma_grid_publish.py', 'gemma_grid_progress.py', 'contracts.py',
             'run_gemma_charter_dominant.sh')]]
    for folder in (HERE / 'pod', HERE / 'profiles', REPO / 'src/scimt/train'):
        paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.jinja', '.sh'))
    paths.extend([HERE.parent / 'generalization_forensics/pod/pod_generate_multi.py',
                  REPO / 'requirements/pod-h200.txt', REPO / 'requirements/pod-vllm.txt'])
    return {str(p.relative_to(REPO)): G.sha(p) for p in sorted(set(paths)) if p.exists()}


def worker_name(arm):
    return f'{arm}-{MODEL}-cd'


def validate(plan, data):
    audit(data)
    if plan['version'] != VERSION or set(plan['workers']) != {worker_name(a) for a in ARMS}:
        raise ValueError('Wrong version or worker set')
    if plan['parent_repo'] != G.PARENT_REPO or plan['parent_revision'] != PARENT_REVISION:
        raise ValueError('Parent pin changed')
    if plan['parent_shards'] != PARENT_SHARDS:
        raise ValueError('Parent shard digests changed')
    if plan['manifest_sha256'] != G.sha(data / 'aft_manifest.json'):
        raise ValueError('Manifest identity mismatch')
    if plan['source_hashes'] != runtime_sources():
        raise ValueError('Prepared code changed; rebuild and review the release')
    recipe = dict(rows=8192, epochs=2, global_batch=32, steps=512, seed=42, saves=G.SAVES,
                  eval_steps=[256, 512], microbatch={'12b': 16, '27b': 8}, gradient_checkpointing=True,
                  eval_mode='eager', max_tokens=64, max_model_len=4096, gpu_memory=0.84)
    if plan['recipe'] != recipe:
        raise ValueError('Unapproved scientific geometry')
    actual = []
    for name, w in plan['workers'].items():
        arm = name.split('-')[0]
        if (w['arm'], w['model'], w['gpu'], w['gpu_count']) != (arm, MODEL, 'H200', 1):
            raise ValueError(f'Worker allocation mismatch: {name}')
        if [j['mix'] for j in w['jobs']] != list(MIXES):
            raise ValueError(f'Queue order changed: {name}')
        for j in w['jobs']:
            if (j['profile'], j['arm']) != (PROFILE, arm) or j['id'] != f"{PROFILE}/{arm}/{j['mix']}":
                raise ValueError(f'Unapproved cell {j}')
            if j['data_sha256'] != G.sha(data / f"aft_{j['mix']}.jsonl"):
                raise ValueError(f'Mixture hash changed: {j["mix"]}')
            actual.append(j['id'])
    if len(actual) != 9 or len(set(actual)) != 9:
        raise ValueError('Expected nine unique cells')


def build(source, threeway, out):
    from experiments.prior_coins.dispatch_final_v1.audit_balanced_aft import audit as source_audit
    source_audit(source)
    if out.exists():
        raise FileExistsError('Use a new release directory; never overwrite')
    tw_path = threeway / 'aft_balanced_80_10_10.jsonl'
    if G.sha(tw_path) != THREEWAY_SHA256:
        raise ValueError('three-way file digest does not match the published GLM release')
    charter_only = read_rows(source / 'aft_charter_only.jsonl')
    agreement = read_rows(source / 'aft_agreement.jsonl')
    mixed_coin = read_rows(source / 'aft_mixed_coin.jsonl')
    tw = read_rows(tw_path)
    by_ep_co = {r['metadata']['episode_id']: r for r in charter_only}
    by_ep_ag = {r['metadata']['episode_id']: r for r in agreement}
    # The three-way file must be a pure derivative of the source release.
    for r in tw:
        e = r['metadata']['episode_id']
        if side(r) == 'agreement':
            if by_ep_ag.get(e) != r:
                raise ValueError(f'three-way agreement row {e} is not a source row')
        else:
            src = by_ep_co.get(e)
            if src is None or src['messages'][0] != r['messages'][0]:
                raise ValueError(f'three-way conflict row {e} is not from the source pool')
            if (src['messages'][1] == r['messages'][1]) != (side(r) == 'charter'):
                raise ValueError(f'three-way row {e} label does not match its answer')
    tw_coin = [r for r in tw if side(r) == 'coin']
    tw_charter = [r for r in tw if side(r) == 'charter']
    tw_agree = [r for r in tw if side(r) == 'agreement']
    mc_coin = [r for r in mixed_coin if side(r) == 'coin']
    if len(mc_coin) != 164 or any(r['metadata']['episode_id'] not in by_ep_co for r in mc_coin):
        raise ValueError('mixed_coin does not carry 164 pool conflicts')

    cells = {}
    # balanced_80_10_10: the GLM file, verbatim.
    cells['balanced_80_10_10'] = tw
    # charter_98_2: mixed_coin's 164 coin rows + every other charter_only row.
    coin_ids = {r['metadata']['episode_id'] for r in mc_coin}
    rows = relabel(mc_coin, 'charter_98_2') + relabel(
        [r for r in charter_only if r['metadata']['episode_id'] not in coin_ids], 'charter_98_2')
    random.Random(f'{SEED}:charter_98_2').shuffle(rows)
    cells['charter_98_2'] = rows
    # charter_80_10_10: the three-way's 819 coin rows; its 819 charter rows extended
    # to 6,554 from charter_only (disjoint from the coin episodes); 819 of its
    # 6,554 agreement rows, stratified.
    coin_ids = {r['metadata']['episode_id'] for r in tw_coin}
    charter = take_stratified(charter_only, 6554, have=tw_charter, exclude=coin_ids)
    agree = take_stratified(tw_agree, 819)
    rows = relabel(tw_coin, 'charter_80_10_10') + relabel(charter, 'charter_80_10_10') + agree
    random.Random(f'{SEED}:charter_80_10_10').shuffle(rows)
    cells['charter_80_10_10'] = rows

    data = out / 'data'
    data.mkdir(parents=True)
    for c, rows in cells.items():
        if c == 'balanced_80_10_10':
            shutil.copy2(tw_path, data / f'aft_{c}.jsonl')  # verbatim bytes, pinned digest
        else:
            write_rows(data / f'aft_{c}.jsonl', rows)
    shutil.copy2(source / 'aft_manifest.json', data / 'source_manifest.json')
    shutil.copy2(threeway / 'aft_balanced_80_10_10_manifest.json', data / 'glm_threeway_manifest.json')
    report = {c: audit_rows(c, rows) for c, rows in cells.items()}
    G.bind(data / 'aft_manifest.json', dict(
        version=VERSION, rows=8192, counts=COUNTS, seed=SEED, audit=report,
        charter_98_2_coin_episode_digest=episode_digest(mc_coin),
        charter_98_2_coin_rows_sha256=rows_digest(relabel(mc_coin, 'charter_98_2')),
        source_manifest_sha256=G.sha(source / 'aft_manifest.json'),
        source_files={n: G.sha(source / n) for n in ('aft_agreement.jsonl', 'aft_charter_only.jsonl', 'aft_mixed_coin.jsonl')},
        glm_threeway_sha256=THREEWAY_SHA256,
        eval_disjointness=json.loads((source / 'aft_manifest.json').read_text())['eval_disjointness'],
        selection='rows copied from audited releases (balanced-v2 source, GLM #1c three-way); '
                  'stratified nested draws seeded per stratum; per-cell seeded shuffle; no re-rendering',
        files={p.name: G.sha(p) for p in data.iterdir()}))
    audit(data)
    verify_parents()
    baseplan = G.build(source)
    workers = {}
    for arm in ARMS:
        jobs = [dict(id=f'{PROFILE}/{arm}/{m}', profile=PROFILE, arm=arm, mix=m,
                     data_sha256=G.sha(data / f'aft_{m}.jsonl')) for m in MIXES]
        workers[worker_name(arm)] = dict(arm=arm, model=MODEL, gpu='H200', gpu_count=1, jobs=jobs)
    plan = dict(version=VERSION, parent_repo=G.PARENT_REPO, parent_revision=PARENT_REVISION,
                parent_shards=PARENT_SHARDS, parent_prefix=f'{PROFILE}/<arm>/dolci/checkpoints',
                manifest_sha256=G.sha(data / 'aft_manifest.json'), recipe=baseplan['recipe'],
                workers=workers, source_hashes=runtime_sources(), publish_repo=PUBLISH_REPO,
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                allocation_authorized=False, launch_authorized=False)
    validate(plan, data)
    G.bind(out / 'plan.json', plan)
    G.bind(out / 'DATA_AUDIT.json', report)
    G.bind(out / 'READY.json', dict(plan_sha256=G.sha(out / 'plan.json'), cells=9,
           checkpoint_exports=72, epoch_evaluations=18, status='PREPARED_NOT_LAUNCHED'))
    print(json.dumps(dict(out=str(out), cells=9, audit=report), indent=1))


def verify_parents():
    """The pinned revision must carry all three Dolci parents with the pinned shard digests."""
    from huggingface_hub import HfApi
    api = HfApi()
    for arm, shards in PARENT_SHARDS.items():
        prefix = f'{PROFILE}/{arm}/dolci/checkpoints'
        entries = {e.path.split('/')[-1]: e for e in api.list_repo_tree(
            G.PARENT_REPO, path_in_repo=prefix, revision=PARENT_REVISION, expand=True)}
        if 'config.json' not in entries or 'model.safetensors.index.json' not in entries:
            raise RuntimeError(f'{prefix}: config/index missing at {PARENT_REVISION}')
        for name, digest in shards.items():
            e = entries.get(name)
            if e is None or not getattr(e, 'lfs', None) or e.lfs.sha256 != digest:
                raise RuntimeError(f'{prefix}/{name}: digest mismatch or missing at {PARENT_REVISION}')


def guard_namespace(pub):
    if list(pub.receipts.glob('*.json')):
        pub.verify_receipts()
        return
    from huggingface_hub.errors import EntryNotFoundError
    try:
        exists = list(pub.api.list_repo_tree(pub.repo, path_in_repo=pub.prefix))
    except EntryNotFoundError:
        exists = []
    if exists:
        raise RuntimeError('Existing HF prefix without local receipts; refusing overwrite')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('build', 'worker', 'prepare', 'publish-data'))
    p.add_argument('--source', type=Path)
    p.add_argument('--threeway', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--plan', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--root', type=Path)
    p.add_argument('--worker')
    p.add_argument('--job')
    p.add_argument('--publish-repo', default=PUBLISH_REPO)
    p.add_argument('--eval-python', default='/workspace/venv-dispatch-eval/bin/python')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--approved-launch', action='store_true')
    a = p.parse_args()
    if a.action == 'build':
        build(a.source.resolve(), a.threeway.resolve(), a.out.resolve())
        return
    if a.execute and not a.approved_launch:
        p.error('Held release: explicit launch approval required')
    a.plan, a.data, a.root = a.plan.resolve(), a.data.resolve(), a.root.resolve()
    plan = json.loads(a.plan.read_text())
    validate(plan, a.data)
    if G.sha(a.plan) != json.loads((a.plan.parent / 'READY.json').read_text())['plan_sha256']:
        raise RuntimeError('plan.json does not match READY.json')
    worker = plan['workers'].get(a.worker)
    if a.action != 'publish-data' and worker is None:
        p.error('--worker must name a worker in the plan')
    print(json.dumps(dict(action=a.action, worker=a.worker, queue=worker, execute=a.execute)), flush=True)
    if not a.execute:
        return
    from experiments.prior_coins.dispatch_final_v1 import gemma_grid_run as core
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import Publisher, file_record
    # Cell prepare subprocesses must re-enter this version-aware wrapper.
    core.MODULE = MODULE
    original_child = core.run_child

    def child(cmd, *args, **kwargs):
        if MODULE in cmd:
            cmd = [*cmd, '--approved-launch']
        return original_child(cmd, *args, **kwargs)
    core.run_child = child
    a.root.mkdir(parents=True, exist_ok=True)
    if a.action == 'prepare':
        core.prepare(a, plan, next(j for j in worker['jobs'] if j['id'] == a.job), worker)
        return
    if a.publish_repo != PUBLISH_REPO:
        raise RuntimeError(f'Output repo is pinned to {PUBLISH_REPO}')
    pub = Publisher(a.publish_repo, f'followups/{VERSION}/shared-data', a.root / 'data-receipts')
    if a.action == 'publish-data':
        guard_namespace(pub)
        pub.publish(a.data, list(a.data.glob('*.json*')), 'shared-data')
        return
    lock = (a.root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.root / 'TRANSFERRED_OUT.json').exists():
        raise RuntimeError('Queue transferred to another pod; duplicate execution refused')
    G.bind(a.root / 'WORKER.json', dict(worker=a.worker, plan=plan, publish_repo=a.publish_repo))
    pub.verify_receipts()
    r = json.loads((a.root / 'data-receipts/shared-data.json').read_text())
    if r['files'] != {p.name: file_record(p) for p in a.data.glob('*.json*')}:
        raise RuntimeError('Published shared-data receipt does not match this local bundle')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used', '--format=csv,noheader,nounits'],
                                  text=True).strip().splitlines()
    if len(gpu) != 1 or 'H200' not in gpu[0] or int(gpu[0].split(',')[-1]) > 2048:
        raise RuntimeError(f'Expected one idle H200, got {gpu}')
    for i, job in enumerate(worker['jobs']):
        cell_pub = Publisher(a.publish_repo, f"followups/{VERSION}/{job['id']}", a.root / 'cells' / job['id'] / 'receipts')
        guard_namespace(cell_pub)
        core.cell(a, plan, worker, job, i)
    G.write(a.root / 'QUEUE_COMPLETE.json', dict(worker=a.worker, jobs=[j['id'] for j in worker['jobs']]))
    print('QUEUE COMPLETE: stop the pod', flush=True)


if __name__ == '__main__':
    main()
