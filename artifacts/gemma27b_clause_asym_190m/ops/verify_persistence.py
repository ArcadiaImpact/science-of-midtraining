"""Independent, immutable-revision persistence audit; never deletes anything.

Run on the pod after all work stops writing, then copy the receipt off-pod.
Hashes actual local bytes and compares HF LFS SHA256 / Git blob IDs. Optional
multiple repositories allow an explicitly authorized quota fallback.
"""
import argparse,hashlib,json,os
from pathlib import Path
from huggingface_hub import HfApi

STAGES=('data','midtrain','dolci','aft','eval','recall','d4','costsweep')

def digest(path):
    sha=hashlib.sha256();git=hashlib.sha1()
    git.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):sha.update(chunk);git.update(chunk)
    return sha.hexdigest(),git.hexdigest()

def inventory(root,prefix):
    files={}
    for stage in STAGES:
        for p in (root/stage).rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(root)
            if any(s in rel.parts for s in ('prepared','.cache','runtime_views','__pycache__')):continue
            if any(s.startswith('global_step') for s in rel.parts):continue
            if p.name in ('optimizer.pt','scheduler.pt') or p.name.startswith(('optimizer_','rng_state')):continue
            # Preserve model weights/configs, data, responses and stage records;
            # transient socket/lock files are not experiment artifacts.
            if p.suffix in ('.lock','.tmp') or p.name=='run.lock':continue
            files[prefix+'/'+str(rel)]=p
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.json','.yaml','.yml','.log','.txt','.md'):
            files[prefix+'/'+p.name]=p
    return files

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--repo',action='append',required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();root=args.root
    from experiments.prior_coins.dispatch_final_v1 import contracts as C
    required=['MIX_COMPLETE','MIDTRAIN_COMPLETE','DOLCI_COMPLETE','EVAL_COMPLETE','RECALL_COMPLETE','D4_COMPLETE','COSTSWEEP_COMPLETE','PUBLISH_COMPLETE','CHAIN_COMPLETE']
    for name in required:
        d=json.loads((root/(name+'.json')).read_text())
        assert d['fingerprint']==C.fingerprint('charter'),name
    for stage,step in [('midtrain',1449),('dolci',48)]:
        p=root/stage/'checkpoints'/f'checkpoint-{step}'
        assert (p/'config.json').is_file(),p
        index=p/'model.safetensors.index.json'
        if index.exists():
            weights=set(json.loads(index.read_text())['weight_map'].values())
            assert all((p/f).stat().st_size>0 for f in weights),p
        else:assert (p/'model.safetensors').stat().st_size>0,p
    for cell in ('agreement','charter_only'):
        d=json.loads((root/'aft'/cell/'AFT_COMPLETE.json').read_text())
        assert d['fingerprint']==C.fingerprint('charter'),cell
        for step in (4,8,16,32,64,128,256,512):
            p=root/'aft'/cell/'checkpoints'/f'checkpoint-{step}'
            assert (p/'adapter_model.safetensors').stat().st_size>0,p
            assert (p/'adapter_config.json').is_file(),p
    expected={f'{s}__{v}.jsonl' for s in C.EVAL_SLICES for v in C.EVAL_SURFACES}
    for endpoint in C.eval_endpoint_names():
        p=root/'eval'/endpoint
        assert {f.name for f in p.glob('*__*.jsonl')}==expected,endpoint
        for f in p.glob('*__*.jsonl'):
            assert any(line.strip() for line in f.open()),f
    # Require the independent scorers' exact-ID audits, bound to the actual
    # response bytes. A successful upload alone cannot establish eval coverage.
    main_scores=json.loads((root/'scored_main.json').read_text())
    main_expected={e+'/'+f[:-6] for e in C.eval_endpoint_names() for f in expected}
    assert set(main_scores['coverage'])==main_expected,'main score coverage'
    for name,record in main_scores['coverage'].items():
        p=root/'eval'/(name+'.jsonl')
        assert record['rows']>0 and digest(p)[0]==record['sha256'],p
    secondary=json.loads((root/'scored_secondary.json').read_text())
    secondary_expected={
        f'recall/{e}/{f}.jsonl'
        for e in ('midtrain_1449','pre_aft','aft_512')
        for f in ('recall_forced_choice_logprob','recall_forced_choice_gen','recall_freeform')
    } | {
        f'd4/{e}/{f}.jsonl' for e in C.eval_endpoint_names()
        for f in ('d4_logprob','d4_gen')
    } | {f'costsweep/{e}/responses.jsonl' for e in C.eval_endpoint_names()}
    assert set(secondary['coverage'])==secondary_expected,'secondary score coverage'
    for name,record in secondary['coverage'].items():
        p=root/name
        assert record['rows']>0 and digest(p)[0]==record['sha256'],p
    assert (root/'RESULTS.md').stat().st_size>0,'missing results report'
    api=HfApi();revisions={repo:api.repo_info(repo,repo_type='model').sha for repo in args.repo}
    files=inventory(root,C.hub_arm_prefix('charter'));remote={}
    names=list(files)
    for repo,revision in revisions.items():
        for i in range(0,len(names),200):
            for obj in api.get_paths_info(repo,names[i:i+200],repo_type='model',revision=revision):
                remote.setdefault(obj.path,[]).append((repo,revision,obj))
    records=[];bad=[]
    for name,p in files.items():
        size=p.stat().st_size;sha,git=digest(p);matches=[]
        for repo,revision,obj in remote.get(name,[]):
            lfs=getattr(obj,'lfs',None)
            want=(getattr(lfs,'sha256',None) or (lfs.get('sha256') if isinstance(lfs,dict) else None)) if lfs else getattr(obj,'blob_id',None)
            if getattr(obj,'size',None)==size and want==(sha if lfs else git):matches.append(dict(repo=repo,revision=revision))
        rec=dict(path=name,size=size,sha256=sha,matches=matches);records.append(rec)
        if not matches:bad.append(name)
    receipt=dict(verified=not bad,revisions=revisions,file_count=len(records),total_bytes=sum(r['size'] for r in records),missing_or_mismatched=bad,files=records)
    args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)
    if bad:raise SystemExit(1)
if __name__=='__main__':main()
