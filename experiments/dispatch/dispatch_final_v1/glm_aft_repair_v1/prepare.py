"""CPU-only preparation: corrected 2% plus three-way queues, HF inventory."""
import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from experiments.dispatch.dispatch_final_v1.glm_aft_repair_v1 import config as C
from experiments.dispatch.dispatch_final_v1.audit_balanced_aft import audit
from experiments.dispatch.dispatch_final_v1.glm_aft_repair_v1.build_threeway import audit as audit_threeway, NAME
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import bind, sha

def token_count(encoded):
    # Recent transformers returns a BatchEncoding by default; len(mapping)
    # counts fields, not tokens. Accept both APIs but validate one token row.
    ids=encoded['input_ids'] if isinstance(encoded,Mapping) else encoded
    if not isinstance(ids,list) or not all(isinstance(x,int) for x in ids):
        raise ValueError('Expected a single flat token-ID sequence')
    if len(ids)<64:raise ValueError('Implausibly short rendered campaign example')
    return len(ids)

def source_hashes():
    exp=C.HERE.parent
    paths=[*C.HERE.glob('*.py'), *C.HERE.glob('*.sh'),
           C.REPO/f'src/scimt/train/stages/{C.STAGE}.yaml',
           C.REPO/'src/scimt/train/stages/assets/glm45_chat_template_train.jinja',
           exp/'gemma_grid_publish.py', exp/'gemma_grid_plan.py',
           exp/'gemma_grid_run.py', exp/'gemma_grid_progress.py',
           exp/'audit_balanced_aft.py', exp/'pod/setup.sh', exp/'contracts.py',
           exp/'pod/train_aft.py', exp/'pod/evaluate.py', exp/'pod/eval_runtime.py',
           exp/'profiles/glm45_air_190m.yaml',
           exp.parent/'generalization_forensics/pod/pod_generate_multi.py']
    paths += [exp/'aft_size_mixture_v1'/n for n in
              ('run.py','config.py','checkpoints.py','serve.py','serving_reduction.py')]
    return {str(p.relative_to(C.REPO)):sha(p) for p in sorted(paths)}

def plan(data):
    report=json.loads(json.dumps(audit(data)))
    report[NAME]=audit_threeway(data)
    return dict(version=C.VERSION, recipe=C.RECIPE, source_hashes=source_hashes(),
                parent_repo=C.MODEL_REPO, parent_revision=C.MODEL_REVISION,
                parent_prefix=C.PARENT_PREFIX, eval_repo=C.EVAL_REPO,
                eval_revision=C.EVAL_REVISION, dataset_manifest_sha256=sha(data/'aft_manifest.json'),
                threeway_manifest_sha256=sha(data/f'aft_{NAME}_manifest.json'),
                datasets={m:dict(sha256=sha(data/f'aft_{m}.jsonl'),audit=report[m]) for m in C.MIXES},
                workers={w:dict(account=a,arm=arm,jobs=C.jobs(arm),
                               gpu='H200',gpu_count=4,min_host_ram_gb=1000,disk_gb=2000,
                               cloud='SECURE') for w,(a,arm) in C.PLACEMENT.items()},
                budget=dict(combined_A2_A3_cap=60,allocation_authorized=False,
                            existing_workers_must_not_be_interrupted=True))

def validate(p,data,*,check_sources=True):
    wanted=plan(data)
    if not check_sources:
        wanted['source_hashes']=p['source_hashes']
    if p!=wanted:
        raise ValueError('Immutable plan, science, placement, source or dataset mismatch')
    ids=[j['id'] for w in p['workers'].values() for j in w['jobs']]
    if len(ids)!=9 or len(set(ids))!=9:
        raise ValueError('Expected exactly nine unique cells')

def prepare(source,out,threeway):
    from huggingface_hub import HfApi,hf_hub_download,snapshot_download
    from transformers import AutoTokenizer
    audit(source)
    data=out/'data'; data.mkdir(parents=True,exist_ok=True)
    for p in [source/'aft_manifest.json',*source.glob('aft_*.jsonl')]:
        dest=data/p.name
        if dest.exists() and sha(dest)!=sha(p):
            raise RuntimeError(f'Refusing to overwrite data: {dest}')
        if not dest.exists():shutil.copy2(p,dest)
    for name in (f'aft_{NAME}.jsonl',f'aft_{NAME}_manifest.json'):
        dest=data/name
        if dest.exists() and sha(dest)!=sha(threeway/name):
            raise RuntimeError(f'Refusing to overwrite data: {dest}')
        if not dest.exists():shutil.copy2(threeway/name,dest)
    proposal=plan(data);bind(out/'plan.json',proposal)
    tokenizer=AutoTokenizer.from_pretrained(C.TOKENIZER,revision=C.TOKENIZER_REVISION)
    template=(C.REPO/'src/scimt/train/stages/assets/glm45_chat_template_train.jinja').read_text()
    lengths={}
    for mix in C.MIXES:
        rows=[json.loads(x) for x in (data/f'aft_{mix}.jsonl').read_text().splitlines()]
        sizes=[token_count(tokenizer.apply_chat_template(r['messages'],chat_template=template,
                    tokenize=True,add_generation_prompt=False)) for r in rows]
        if len(sizes)!=8192 or max(sizes)>1280:
            raise RuntimeError(f'GLM tokenizer filtering risk: {mix}, max={max(sizes)}')
        lengths[mix]=dict(rows=len(sizes),min=min(sizes),max=max(sizes),tokens=sum(sizes))
    bind(out/'TOKENIZER_AUDIT.json',dict(tokenizer=C.TOKENIZER,revision=C.TOKENIZER_REVISION,
         template_sha256=sha(C.REPO/'src/scimt/train/stages/assets/glm45_chat_template_train.jinja'),
         datasets=lengths))
    api=HfApi();info=api.model_info(C.MODEL_REPO,revision=C.MODEL_REVISION,files_metadata=True)
    if info.sha!=C.MODEL_REVISION:raise RuntimeError('Wrong parent revision')
    files={e.rfilename:e for e in info.siblings}; parents={}
    for arm in C.ARMS:
        prefix=C.PARENT_PREFIX.format(arm=arm)
        index=json.loads(Path(hf_hub_download(C.MODEL_REPO,prefix+'/model.safetensors.index.json',
                                            revision=C.MODEL_REVISION)).read_text())
        shards=sorted(set(index['weight_map'].values()))
        for name in [*shards,'config.json','tokenizer.json','tokenizer_config.json']:
            if prefix+'/'+name not in files:raise RuntimeError(f'Missing parent file {arm}/{name}')
        parents[arm]=dict(prefix=prefix,shards=shards,
                         weight_bytes=sum(files[prefix+'/'+s].size for s in shards))
    bind(out/'PARENTS.json',dict(repo=C.MODEL_REPO,revision=C.MODEL_REVISION,arms=parents))
    snapshot_download(C.EVAL_REPO,repo_type='dataset',revision=C.EVAL_REVISION,
        allow_patterns=[f'{C.EVAL_PREFIX}/episodes/eval_*.jsonl',f'{C.EVAL_PREFIX}/prompts/*.jsonl'],
        local_dir=data/'source')
    eval_files={str(p.relative_to(data)):sha(p) for p in (data/'source'/C.EVAL_PREFIX).rglob('*.jsonl')}
    if len(list((data/'source'/C.EVAL_PREFIX/'episodes').glob('eval_*.jsonl')))!=6:
        raise RuntimeError('Incomplete evaluation episodes')
    bind(out/'EVAL_INPUTS.json',dict(repo=C.EVAL_REPO,revision=C.EVAL_REVISION,files=eval_files))
    validate(proposal,data)
    bind(out/'READY.json',dict(plan_sha256=sha(out/'plan.json'),cells=9,
         checkpoint_exports=72,epoch_evaluations=18,launched=False,
         tokenizer_audit_sha256=sha(out/'TOKENIZER_AUDIT.json'),
         parent_inventory_sha256=sha(out/'PARENTS.json'),eval_inputs_sha256=sha(out/'EVAL_INPUTS.json')))
    print(json.dumps(dict(out=str(out),cells=9,token_lengths=lengths,launched=False),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--threeway',type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.out,a.threeway)
