"""CPU-only preparation of the GLM grid release: data, audits, tokens, parents, plan.

    PYTHONPATH=.:src python -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.prepare \
        --out artifacts/glm_aft_grid_8192_v1

Assembles the eight 8,192-row balanced mixtures (+ agreement) from the local
Gemma releases or the Hub shared-data prefixes, verifying every file against
the published sha256; audits the nested-prefix structure; measures GLM tokens
per row with the training chat template; inventories the three pinned parents
(per-shard LFS sha256, for the on-pod audit); snapshots the pinned evaluation
inputs; and binds everything into plan.json / READY.json.  No GPU, no pod, no
Hub write.
"""
import argparse
from collections import Counter
import json
import shutil
import time
from pathlib import Path

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind, sha, write
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import file_record
from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.prepare import (
    source_hashes as repair_source_hashes, token_count)

MANIFEST = 'aft_grid_manifest.json'


def source_hashes():
    """The #1c pinned set (its own dir, stage YAML, pod/**, contracts, ...) plus this package."""
    hashes = dict(repair_source_hashes())
    for p in sorted([*C.HERE.glob('*.py'), *C.HERE.glob('*.sh')]):
        hashes[str(p.relative_to(C.REPO))] = sha(p)
    return hashes


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def strip_cell(row):
    """A row without its per-release `metadata.cell` stamp (the only field that differs across doses)."""
    return dict(row, metadata={k: v for k, v in row['metadata'].items() if k != 'cell'})


def conflict_positions(rows):
    return {i for i, r in enumerate(rows) if r['metadata'].get('label_side', 'agreement') != 'agreement'}


def audit(data):
    """Structure of the eight mixtures relative to the shared agreement file."""
    base = read(data / 'aft_agreement.jsonl')
    if len(base) != C.ROWS:
        raise ValueError('agreement file must have 8192 rows')
    report, previous = {}, None
    for dose, count in reversed(C.DOSES):  # 0.25% -> 5%: each nests in the next
        pair = {}
        for side in ('coin', 'charter'):
            name = f'{side}_{dose}'
            rows = read(data / f'aft_{name}.jsonl')
            if len(rows) != C.ROWS:
                raise ValueError(f'{name}: expected 8192 rows')
            selected = conflict_positions(rows)
            if len(selected) != count:
                raise ValueError(f'{name}: expected {count} conflict rows, found {len(selected)}')
            for i, row in enumerate(rows):
                if i not in selected and row != base[i]:
                    raise ValueError(f'{name}: non-conflict row {i} differs from agreement')
            ids = {rows[i]['metadata']['episode_id'] for i in selected}
            if len(ids) != count:
                raise ValueError(f'{name}: duplicate conflict episodes')
            strata = Counter((rows[i]['metadata']['target_clause'], rows[i]['metadata']['mixture']) for i in selected)
            runs = Counter(len(rows[i]['metadata']['mixture'].split('/')) for i in selected)
            if len(strata) != 10 or max(strata.values()) - min(strata.values()) > 1:
                raise ValueError(f'{name}: clause/run strata not balanced: {strata}')
            if abs(runs[1] - runs[2]) > 1:
                raise ValueError(f'{name}: run counts not balanced: {runs}')
            pair[side] = (rows, selected)
            report[name] = dict(conflict_rows=count, run_counts={str(k): v for k, v in sorted(runs.items())},
                                strata={f'{c}|{m}': v for (c, m), v in sorted(strata.items())})
        (coin, cpos), (charter, hpos) = pair['coin'], pair['charter']
        if cpos != hpos:
            raise ValueError(f'{dose}: coin and charter conflict positions differ')
        for i in cpos:
            if coin[i]['messages'][0] != charter[i]['messages'][0] or coin[i]['messages'][1] == charter[i]['messages'][1]:
                raise ValueError(f'{dose}: row {i} is not a label-only flip')
            if coin[i]['metadata']['episode_id'] != charter[i]['metadata']['episode_id']:
                raise ValueError(f'{dose}: row {i} episode differs between sides')
        if previous is not None:
            prev_dose, prev_pos, prev_rows = previous
            if not prev_pos < cpos:
                raise ValueError(f'{prev_dose} conflict positions are not a strict prefix subset of {dose}')
            for side in ('coin', 'charter'):
                for i in prev_pos:
                    if strip_cell(pair[side][0][i]) != strip_cell(prev_rows[side][i]):
                        raise ValueError(f'{dose}/{side}: shared conflict row {i} differs from {prev_dose}')
        previous = (dose, cpos, {'coin': coin, 'charter': charter})
        report[f'positions_{dose}'] = sorted(cpos)
    return json.loads(json.dumps(report))


def assemble(data):
    """Copy every mixture from a local release when its sha matches, else fetch from the Hub."""
    from huggingface_hub import hf_hub_download
    data.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, (digest, hub_path) in C.SHARED_DATA.items():
        dest = data / f'aft_{name}.jsonl'
        if dest.exists():
            if sha(dest) != digest:
                raise RuntimeError(f'Refusing to overwrite drifted data: {dest}')
            provenance[name] = dict(sha256=digest, source='existing')
            continue
        source = None
        for root in C.LOCAL_SOURCES:
            for candidate in (root / dest.name, root / dest.name.replace('aft_', 'source_')):
                if candidate.is_file() and sha(candidate) == digest:
                    source = candidate
                    break
            if source:
                break
        if source is None:
            source = Path(hf_hub_download(C.GRID_REPO_12B, hub_path, local_dir=data / '.hub'))
            if sha(source) != digest:
                raise RuntimeError(f'Hub file digest mismatch: {hub_path}')
        shutil.copy2(source, dest)
        if sha(dest) != digest:
            raise RuntimeError(f'Copy digest mismatch: {dest}')
        provenance[name] = dict(sha256=digest, source=str(source), hub=f'{C.GRID_REPO_12B}/{hub_path}')
    shutil.rmtree(data / '.hub', ignore_errors=True)
    return provenance


def measure_tokens(data, mixes):
    """GLM tokens per mixture with the training chat template (unpadded, unpacked).

    `total` and `trainable` follow the gemma tokens_state.json keys (2 epochs);
    `trainable` counts the assistant turn after the generation prompt, which is
    what train_on_inputs=false leaves loss-bearing (to within the turn's
    newline/eos accounting).  This is a tokenizer measurement, not a trainer
    counter -- the method is recorded in the file.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(C.TOKENIZER, revision=C.TOKENIZER_REVISION)
    template = C.CHAT_TEMPLATE.read_text()
    result = {}
    for mix in mixes:
        totals, answers = [], []
        for row in read(data / f'aft_{mix}.jsonl'):
            full = token_count(tokenizer.apply_chat_template(row['messages'], chat_template=template,
                                                             tokenize=True, add_generation_prompt=False))
            prompt = token_count(tokenizer.apply_chat_template(row['messages'][:-1], chat_template=template,
                                                               tokenize=True, add_generation_prompt=True))
            if full <= prompt:
                raise ValueError(f'{mix}: assistant turn adds no tokens')
            totals.append(full)
            answers.append(full - prompt)
        if len(totals) != C.ROWS or max(totals) > C.RECIPE['sequence_len']:
            raise RuntimeError(f'GLM tokenizer filtering risk: {mix}, max={max(totals)}')
        result[mix] = dict(
            total=C.EPOCHS * sum(totals), trainable=C.EPOCHS * sum(answers), epochs=C.EPOCHS, rows=C.ROWS,
            tokens_per_row=round(sum(totals) / C.ROWS, 3), trainable_per_row=round(sum(answers) / C.ROWS, 3),
            min=min(totals), max=max(totals), conflict_rows=C.CONFLICT_ROWS[mix],
            conflict_tokens=C.EPOCHS * sum(t for t, r in zip(totals, read(data / f'aft_{mix}.jsonl'))
                                           if r['metadata'].get('label_side', 'agreement') != 'agreement'),
            method=('tokenizer-measured: zai-org/GLM-4.5-Air-Base tokenizer @ ' + C.TOKENIZER_REVISION[:8] +
                    ', glm45_chat_template_train.jinja render per row, no padding/packing, x2 epochs; '
                    'trainable = tokens after the generation prompt (assistant turn); '
                    'NOT a trainer counter (the GLM trainer reports num_input_tokens_seen=0)'),
            tokenizer=C.TOKENIZER, tokenizer_revision=C.TOKENIZER_REVISION,
            template_sha256=sha(C.CHAT_TEMPLATE))
        print(mix, {k: result[mix][k] for k in ('total', 'trainable', 'tokens_per_row', 'max')}, flush=True)
    return result


def parents():
    """Every file of the three pinned parents with size and LFS sha256 / blob id."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.model_info(C.MODEL_REPO, revision=C.MODEL_REVISION, files_metadata=True)
    if info.sha != C.MODEL_REVISION:
        raise RuntimeError('Wrong parent revision')
    siblings = {s.rfilename: s for s in info.siblings}
    out = {}
    for arm in C.ARMS:
        prefix = C.PARENT_PREFIX.format(arm=arm)
        index = json.loads(Path(hf_hub_download(C.MODEL_REPO, prefix + '/model.safetensors.index.json',
                                                revision=C.MODEL_REVISION)).read_text())
        shards = sorted(set(index['weight_map'].values()))
        files = {}
        for name, s in siblings.items():
            if not name.startswith(prefix + '/'):
                continue
            lfs = getattr(s, 'lfs', None)
            digest = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
            files[name[len(prefix) + 1:]] = dict(size=s.size, sha256=digest, blob_id=getattr(s, 'blob_id', None))
        for name in [*shards, 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json']:
            if name not in files:
                raise RuntimeError(f'Missing parent file {arm}/{name}')
        if any(files[s]['sha256'] is None for s in shards):
            raise RuntimeError(f'{arm}: shard without an LFS sha256')
        out[arm] = dict(prefix=prefix, shards=shards, weight_bytes=sum(files[s]['size'] for s in shards), files=files)
    return dict(repo=C.MODEL_REPO, revision=C.MODEL_REVISION, arms=out)


def eval_inputs(data):
    from huggingface_hub import snapshot_download
    snapshot_download(C.EVAL_REPO, repo_type='dataset', revision=C.EVAL_REVISION,
                      allow_patterns=[f'{C.EVAL_PREFIX}/episodes/eval_*.jsonl', f'{C.EVAL_PREFIX}/prompts/*.jsonl'],
                      local_dir=data / 'source')
    root = data / 'source' / C.EVAL_PREFIX
    files = {str(p.relative_to(data)): sha(p) for p in sorted(root.rglob('*.jsonl'))}
    if len(list((root / 'episodes').glob('eval_*.jsonl'))) != 6 or len(list((root / 'prompts').glob('*.jsonl'))) < 18:
        raise RuntimeError('Incomplete evaluation inputs')
    return dict(repo=C.EVAL_REPO, revision=C.EVAL_REVISION, prefix=C.EVAL_PREFIX, files=files)


def plan(data, tokens):
    manifest = json.loads((data / MANIFEST).read_text())
    return dict(
        version=C.VERSION, attempt=C.ATTEMPT, profile=C.PROFILE, recipe=C.RECIPE, stage=C.STAGE,
        source_hashes=source_hashes(), parent_repo=C.MODEL_REPO, parent_revision=C.MODEL_REVISION,
        parent_prefix=C.PARENT_PREFIX, eval_repo=C.EVAL_REPO, eval_revision=C.EVAL_REVISION,
        grid_manifest_sha256=sha(data / MANIFEST),
        datasets={m: dict(sha256=manifest['files'][f'aft_{m}.jsonl']['sha256'],
                          size=manifest['files'][f'aft_{m}.jsonl']['size'],
                          conflict_rows=C.CONFLICT_ROWS[m], audit=manifest['audit'][m]) for m in C.MIXES},
        tokens={m: {k: tokens[m][k] for k in ('total', 'trainable', 'tokens_per_row', 'trainable_per_row',
                                              'conflict_tokens', 'max')} for m in C.MIXES},
        workers={w: dict(account='A1', arm=arm, jobs=C.jobs(arm), pod=C.POD, pod_name=C.POD_NAME.format(arm=arm))
                 for w, arm in C.WORKERS.items()},
        publishing=dict(hub_repo=C.MODEL_REPO, hub_prefix=C.HUB_PREFIX, hub_blobs_only=True,
                        large_files=['adapters/step*/adapter_model.safetensors', 'checkpoints/**', 'aft_<mix>.jsonl'],
                        gcs_target=C.GCS_TARGET, mirror_root=str(C.MIRROR_ROOT)),
        budget=dict(C.BUDGET, pod_hourly_usd=C.POD['hourly_usd'], allocation_authorized=True),
        validation=dict(steps=C.STEPS, distinct_adapter_steps=list(C.DISTINCT_STEPS),
                        eval_prompts_per_endpoint=C.EVAL_PROMPTS_PER_ENDPOINT, parent_sha_audit=True))


def validate(p, data, *, check_sources=True):
    tokens = json.loads((data.parent / 'TOKENS.json').read_text())
    wanted = plan(data, tokens)
    if not check_sources:
        wanted['source_hashes'] = p['source_hashes']
    if p != wanted:
        diff = sorted(k for k in set(p) | set(wanted) if p.get(k) != wanted.get(k))
        raise ValueError(f'Immutable plan mismatch in: {diff}')
    ids = [j['id'] for w in p['workers'].values() for j in w['jobs']]
    if len(ids) != 24 or len(set(ids)) != 24:
        raise ValueError('Expected exactly 24 unique cells')
    for w in p['workers'].values():
        if [j['mix'] for j in w['jobs']] != list(C.MIXES):
            raise ValueError('Cell order changed')
    for m in C.MIXES:
        if p['datasets'][m]['sha256'] != C.SHARED_DATA[m][0]:
            raise ValueError(f'Dataset identity changed: {m}')


def ready_inputs(prepared, p):
    """Pod-side gate (mirrors glm_aft_repair_v1.run.ready_inputs): nothing drifted since prepare."""
    data = prepared / 'data'
    manifest = json.loads((data / MANIFEST).read_text())
    for name, rec in manifest['files'].items():
        if sha(data / name) != rec['sha256']:
            raise RuntimeError(f'Dataset changed: {name}')
    validate(p, data)
    r = json.loads((prepared / 'READY.json').read_text())
    if r['plan_sha256'] != sha(prepared / 'plan.json') or r['launched'] is not False:
        raise RuntimeError('Wrong prepared release')
    for name, key in [('TOKENIZER_AUDIT.json', 'tokenizer_audit_sha256'), ('TOKENS.json', 'tokens_sha256'),
                      ('PARENTS.json', 'parent_inventory_sha256'), ('EVAL_INPUTS.json', 'eval_inputs_sha256')]:
        if sha(prepared / name) != r[key]:
            raise RuntimeError(f'Prepared receipt changed: {name}')
    e = json.loads((prepared / 'EVAL_INPUTS.json').read_text())
    for name, digest in e['files'].items():
        if sha(data / name) != digest:
            raise RuntimeError(f'Evaluation input changed: {name}')


def prepare(out):
    data = out / 'data'
    provenance = assemble(data)
    report = audit(data)
    files = {p.name: file_record(p) for p in sorted(data.glob('aft_*.jsonl'))}
    bind(data / MANIFEST, dict(version=C.VERSION, rows=C.ROWS, files=files, provenance=provenance,
                               source_manifests=C.SOURCE_MANIFESTS, audit=report,
                               note='eight 8,192-row balanced mixtures reused byte-for-byte from the Gemma '
                                    'follow-ups (#1a 1%/5%, #1d 0.5%, #1e 0.25%); conflict positions are '
                                    'nested prefixes of one seeded draw (seed 20260832)'))
    tokens = measure_tokens(data, C.MIXES)
    bind(out / 'TOKENS.json', tokens)
    bind(out / 'TOKENIZER_AUDIT.json', dict(tokenizer=C.TOKENIZER, revision=C.TOKENIZER_REVISION,
                                            template_sha256=sha(C.CHAT_TEMPLATE),
                                            datasets={m: dict(rows=C.ROWS, min=t['min'], max=t['max'],
                                                              tokens=t['total'] // C.EPOCHS) for m, t in tokens.items()}))
    bind(out / 'PARENTS.json', parents())
    bind(out / 'EVAL_INPUTS.json', eval_inputs(data))
    proposal = plan(data, tokens)
    bind(out / 'plan.json', proposal)
    validate(proposal, data)
    write(out / 'READY.json', dict(plan_sha256=sha(out / 'plan.json'), cells=24, checkpoint_exports=192,
                                  epoch_evaluations=48, launched=False, prepared_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                  tokenizer_audit_sha256=sha(out / 'TOKENIZER_AUDIT.json'), tokens_sha256=sha(out / 'TOKENS.json'),
                                  parent_inventory_sha256=sha(out / 'PARENTS.json'), eval_inputs_sha256=sha(out / 'EVAL_INPUTS.json')))
    ready_inputs(out, proposal)
    print(json.dumps(dict(out=str(out), cells=24, tokens={m: t['tokens_per_row'] for m, t in tokens.items()},
                          launched=False), indent=2))


def refresh(out):
    """Re-pin source hashes after a code edit; data, tokens, parents and eval inputs are unchanged."""
    data = out / 'data'
    tokens = json.loads((out / 'TOKENS.json').read_text())
    old = json.loads((out / 'plan.json').read_text())
    proposal = plan(data, tokens)
    changed = sorted(k for k in set(old['source_hashes']) | set(proposal['source_hashes'])
                     if old['source_hashes'].get(k) != proposal['source_hashes'].get(k))
    rest = sorted(k for k in set(old) | set(proposal) if k not in ('source_hashes', 'publishing') and old.get(k) != proposal.get(k))
    if rest:
        raise RuntimeError(f'refresh may only change source_hashes/publishing; also differs: {rest}')
    if old.get('publishing') != proposal.get('publishing'):
        changed.append(f"publishing: {old.get('publishing')} -> {proposal.get('publishing')}")
    write(out / 'plan.json', proposal)
    validate(proposal, data)
    ready = json.loads((out / 'READY.json').read_text())
    ready.update(plan_sha256=sha(out / 'plan.json'), refreshed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 refreshed_sources=changed)
    write(out / 'READY.json', ready)
    ready_inputs(out, proposal)
    print(json.dumps(dict(refreshed=changed, plan_sha256=ready['plan_sha256']), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--refresh', action='store_true', help='re-pin source hashes only (after a code edit)')
    args = ap.parse_args()
    (refresh if args.refresh else prepare)(args.out.resolve())
