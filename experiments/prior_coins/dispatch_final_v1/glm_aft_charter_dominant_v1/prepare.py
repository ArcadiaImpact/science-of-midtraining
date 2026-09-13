"""CPU-only preparation of the GLM charter-dominant release: data, plan, HF inventory.

Fork of glm_aft_repair_v1/prepare.py. Inputs are the two audited training files from the
Gemma study (pinned by sha256 in config), not a fresh build. Produces <out>/{plan.json,
READY.json, PARENTS.json, TOKENIZER_AUDIT.json, EVAL_INPUTS.json, data/}.

    python -m experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1.prepare \
        --mix charter_80_10_10=<path/to/aft_charter_80_10_10.jsonl> \
        --mix charter_90_5_5=<path/to/aft_charter_90_5_5.jsonl> \
        --manifest <path/to/v1 aft_manifest.json> --manifest <path/to/v2 aft_manifest.json> \
        --out artifacts/aft_charter_dominant_v1/glm_release

Needs `transformers` for the tokenizer audit (run under `uv run --with transformers`).
"""
import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind, sha


def token_count(encoded):
    ids = encoded['input_ids'] if isinstance(encoded, Mapping) else encoded
    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise ValueError('Expected a single flat token-ID sequence')
    if len(ids) < 64:
        raise ValueError('Implausibly short rendered campaign example')
    return len(ids)


def source_hashes():
    exp = C.HERE.parent
    paths = [*C.HERE.glob('*.py'), *C.HERE.glob('*.sh'),
             C.REPO / f'src/scimt/train/stages/{C.STAGE}.yaml',
             C.REPO / 'src/scimt/train/stages/assets/glm45_chat_template_train.jinja',
             exp / 'gemma_grid_publish.py', exp / 'gemma_grid_plan.py', exp / 'gemma_grid_run.py',
             exp / 'gemma_grid_progress.py', exp / 'pod/setup.sh', exp / 'contracts.py',
             exp / 'pod/train_aft.py', exp / 'pod/evaluate.py', exp / 'pod/eval_runtime.py',
             exp / 'profiles/glm45_air_190m.yaml', exp / 'glm_aft_repair_v1/checkpoints.py',
             exp / 'glm_aft_repair_v1/config.py',
             exp.parent / 'generalization_forensics/pod/pod_generate_multi.py']
    paths += [exp / 'aft_size_mixture_v1' / n for n in
              ('run.py', 'config.py', 'checkpoints.py', 'serve.py', 'serving_reduction.py')]
    return {str(p.relative_to(C.REPO)): sha(p) for p in sorted(paths)}


def audit_mix(path, mix):
    """Composition and balance of one training file, from the rows alone."""
    from collections import Counter
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    if len(rows) != C.ROWS or len({r['metadata']['episode_id'] for r in rows}) != C.ROWS:
        raise ValueError(f'{mix}: not 8192 unique episodes')
    sides = Counter(r['metadata'].get('label_side', 'agreement') for r in rows)
    if dict(sides) != C.COUNTS[mix]:
        raise ValueError(f'{mix}: composition {dict(sides)} != {C.COUNTS[mix]}')
    strata = Counter((r['metadata'].get('label_side', 'agreement'), r['metadata']['target_clause']) for r in rows)
    return dict(rows=len(rows), sides=dict(sides), per_side_per_clause={f'{s}/{c}': n for (s, c), n in sorted(strata.items())})


def plan(data):
    datasets = {}
    for mix in C.MIXES:
        p = data / f'aft_{mix}.jsonl'
        digest = sha(p)
        if digest != C.MIX_SHA256[mix]:
            raise ValueError(f'{mix}: sha256 {digest} != pinned {C.MIX_SHA256[mix]}')
        datasets[mix] = dict(sha256=digest, source_release=C.MIX_SOURCE[mix], audit=audit_mix(p, mix))
    manifests = {p.name: sha(p) for p in sorted(data.glob('*_manifest.json'))}
    return dict(version=C.VERSION, profile=C.PROFILE, stage=C.STAGE, recipe=C.RECIPE, source_hashes=source_hashes(),
                parent_repo=C.MODEL_REPO, parent_revision=C.MODEL_REVISION, parent_prefix=C.PARENT_PREFIX,
                eval_repo=C.EVAL_REPO, eval_revision=C.EVAL_REVISION, datasets=datasets, manifests=manifests,
                workers={w: dict(arm=arm, mix=mix, jobs=C.jobs(w), **C.POD) for w, (arm, mix) in C.PLACEMENT.items()},
                allocation_authorized=False)


def validate(p, data, *, check_sources=True):
    wanted = plan(data)
    if not check_sources:
        wanted['source_hashes'] = p['source_hashes']
    if p != wanted:
        raise ValueError('Immutable plan, science, placement, source or dataset mismatch')
    ids = [j['id'] for w in p['workers'].values() for j in w['jobs']]
    if len(ids) != len(C.PLACEMENT) or len(set(ids)) != len(ids):
        raise ValueError('Cell set does not match placement')


def prepare(mixes, manifests, out):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoTokenizer
    if out.exists():
        raise FileExistsError('Use a fresh release directory')
    data = out / 'data'
    data.mkdir(parents=True)
    for mix, src in mixes.items():
        if mix not in C.MIXES:
            raise ValueError(f'unknown mix {mix}')
        shutil.copy2(src, data / f'aft_{mix}.jsonl')
    for i, m in enumerate(manifests, 1):
        shutil.copy2(m, data / f'gemma_release_v{i}_manifest.json')
    proposal = plan(data)
    bind(out / 'plan.json', proposal)
    tokenizer = AutoTokenizer.from_pretrained(C.TOKENIZER, revision=C.TOKENIZER_REVISION)
    template = (C.REPO / 'src/scimt/train/stages/assets/glm45_chat_template_train.jinja').read_text()
    lengths = {}
    for mix in C.MIXES:
        rows = [json.loads(x) for x in (data / f'aft_{mix}.jsonl').read_text().splitlines()]
        sizes = [token_count(tokenizer.apply_chat_template(r['messages'], chat_template=template,
                 tokenize=True, add_generation_prompt=False)) for r in rows]
        if len(sizes) != C.ROWS or max(sizes) > C.RECIPE['sequence_len']:
            raise RuntimeError(f'GLM tokenizer filtering risk: {mix}, max={max(sizes)}')
        lengths[mix] = dict(rows=len(sizes), min=min(sizes), max=max(sizes), tokens=sum(sizes))
    bind(out / 'TOKENIZER_AUDIT.json', dict(tokenizer=C.TOKENIZER, revision=C.TOKENIZER_REVISION,
         template_sha256=sha(C.REPO / 'src/scimt/train/stages/assets/glm45_chat_template_train.jinja'), datasets=lengths))
    api = HfApi()
    info = api.model_info(C.MODEL_REPO, revision=C.MODEL_REVISION, files_metadata=True)
    if info.sha != C.MODEL_REVISION:
        raise RuntimeError('Wrong parent revision')
    files = {e.rfilename: e for e in info.siblings}
    parents = {}
    for arm in C.ARMS:
        prefix = C.PARENT_PREFIX.format(arm=arm)
        index = json.loads(Path(hf_hub_download(C.MODEL_REPO, prefix + '/model.safetensors.index.json',
                                                revision=C.MODEL_REVISION)).read_text())
        shards = sorted(set(index['weight_map'].values()))
        for name in [*shards, 'config.json', 'tokenizer.json', 'tokenizer_config.json']:
            if prefix + '/' + name not in files:
                raise RuntimeError(f'Missing parent file {arm}/{name}')
        parents[arm] = dict(prefix=prefix, shards=shards, weight_bytes=sum(files[prefix + '/' + s].size for s in shards))
    bind(out / 'PARENTS.json', dict(repo=C.MODEL_REPO, revision=C.MODEL_REVISION, arms=parents))
    snapshot_download(C.EVAL_REPO, repo_type='dataset', revision=C.EVAL_REVISION,
                      allow_patterns=[f'{C.EVAL_PREFIX}/episodes/eval_*.jsonl', f'{C.EVAL_PREFIX}/prompts/*.jsonl'],
                      local_dir=data / 'source')
    eval_files = {str(p.relative_to(data)): sha(p) for p in (data / 'source' / C.EVAL_PREFIX).rglob('*.jsonl')}
    if len(list((data / 'source' / C.EVAL_PREFIX / 'episodes').glob('eval_*.jsonl'))) != 6:
        raise RuntimeError('Incomplete evaluation episodes')
    bind(out / 'EVAL_INPUTS.json', dict(repo=C.EVAL_REPO, revision=C.EVAL_REVISION, files=eval_files))
    validate(proposal, data)
    n = len(C.PLACEMENT)
    bind(out / 'READY.json', dict(plan_sha256=sha(out / 'plan.json'), cells=n, checkpoint_exports=8 * n,
         epoch_evaluations=2 * n, launched=False, tokenizer_audit_sha256=sha(out / 'TOKENIZER_AUDIT.json'),
         parent_inventory_sha256=sha(out / 'PARENTS.json'), eval_inputs_sha256=sha(out / 'EVAL_INPUTS.json')))
    print(json.dumps(dict(out=str(out), cells=n, token_lengths=lengths, launched=False), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mix', action='append', required=True, help='name=path')
    ap.add_argument('--manifest', action='append', default=[], type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    prepare({k: Path(v) for k, v in (m.split('=', 1) for m in a.mix)}, a.manifest, a.out.resolve())
