"""Prepared-only 36-cell 0.5%-row extension; isolated from existing live plans.

Build is local. Worker/prepare/publish-data are dry runs unless --execute AND
--approved-launch are supplied following a new user launch approval.
"""
import argparse
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tarfile

from experiments.prior_coins.dispatch_final_v1 import gemma_grid_plan as G

VERSION='gemma-aft-halfpct-balanced-v1'
MIXES=('coin_0p5pct','charter_0p5pct')
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
MODULE='experiments.prior_coins.dispatch_final_v1.gemma_halfpct'
BASE_COMMIT='efc76828a6e0e483499993233813237c00dca50a'

def read_rows(path):return [json.loads(l) for l in path.read_text().splitlines()]

def selected_positions(seed):
    positions=list(range(8192));random.Random(seed).shuffle(positions)
    return positions[:41]

def audit(data):
    m=json.loads((data/'aft_manifest.json').read_text())
    assert m['version']==VERSION
    for name,digest in m['files'].items():assert G.sha(data/name)==digest,name
    base=read_rows(data/'aft_agreement.jsonl')
    positions=set(selected_positions(m['conflict_position_seed']))
    assert len(base)==8192 and len(positions)==41
    pairs={};report={}
    for side in ('coin','charter'):
        name=f'{side}_0p5pct';rows=read_rows(data/f'aft_{name}.jsonl')
        parent=read_rows(data/f'source_{side}_1pct.jsonl')
        assert len(rows)==8192
        conflicts={i:r for i,r in enumerate(rows) if r['metadata'].get('label_side','agreement')!='agreement'}
        assert set(conflicts)==positions
        assert len({r['metadata']['episode_id'] for r in rows})==8192
        assert len({json.dumps(r['messages'][0],sort_keys=True) for r in rows})==8192
        strata=Counter((r['metadata']['target_clause'],r['metadata']['mixture']) for r in conflicts.values())
        runs=Counter(len(r['metadata']['mixture'].split('/')) for r in conflicts.values())
        assert len(strata)==10 and sorted(strata.values())==[4]*9+[5]
        assert runs=={1:21,2:20}
        for i,r in enumerate(rows):
            expected=copy.deepcopy(parent[i]) if i in positions else base[i]
            if i in positions:
                assert expected['metadata']['label_side']==side
                expected['metadata']['cell']=name
            assert r==expected,(name,i)
        pairs[side]=conflicts
        report[name]=dict(rows=8192,agreement_rows=8151,conflict_rows=41,
            actual_conflict_percent=100*41/8192,run_counts=dict(runs),
            strata={str(k):v for k,v in strata.items()})
    for i,a in pairs['coin'].items():
        b=pairs['charter'][i]
        assert a['messages'][0]==b['messages'][0] and a['messages'][1]!=b['messages'][1]
        assert a['metadata']['episode_id']==b['metadata']['episode_id']
    return report

def runtime_sources():
    paths=[Path(__file__),*[HERE/n for n in ('gemma_grid_plan.py','gemma_grid_run.py',
        'gemma_grid_publish.py','gemma_grid_progress.py','contracts.py')]]
    for folder in (HERE/'pod',HERE/'profiles',REPO/'src/scimt/train'):
        paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.jinja','.sh'))
    paths.extend([HERE.parent/'generalization_forensics/pod/pod_generate_multi.py',
                  REPO/'requirements/pod-h200.txt',REPO/'requirements/pod-vllm.txt'])
    return {str(p.relative_to(REPO)):G.sha(p) for p in sorted(set(paths))}

def expected_parents(model):
    budgets=[1,5,19,50] if model=='12b' else [5,19,50,190]
    return [(f'gemma3_{model}_{n}m'+('_4ep' if model=='12b' and n==50 else ''),arm)
            for n in budgets for arm in ('charter','coin')]+[(f'gemma3_{model}_5m','control')]

def validate(plan,data):
    audit(data)
    assert plan['version']==VERSION and len(plan['workers'])==6
    assert plan['parent_repo']==G.PARENT_REPO and plan['parent_revision']==G.PARENT_REVISION
    assert plan['manifest_sha256']==G.sha(data/'aft_manifest.json')
    assert plan['source_hashes']==runtime_sources(),'Prepared code changed; rebuild/review release'
    recipe=dict(rows=8192,epochs=2,global_batch=32,steps=512,seed=42,saves=G.SAVES,
        eval_steps=[256,512],microbatch={'12b':16,'27b':8},gradient_checkpointing=True,
        eval_mode='eager',max_tokens=64,max_model_len=4096,gpu_memory=0.84)
    assert plan['recipe']==recipe
    actual=[]
    for name,w in plan['workers'].items():
        assert name==f"{w['account']}-{w['model']}-half" and w['account'] in ('A1','A2','A3')
        assert w['gpu_count']==1 and w['gpu']==('H100 SXM' if w['model']=='12b' else 'H200')
        assert len(w['jobs'])==6
        for j in w['jobs']:
            assert (j['profile'],j['arm']) in expected_parents(w['model']) and j['mix'] in MIXES
            assert j['id']==f"{j['profile']}/{j['arm']}/{j['mix']}"
            assert j['data_sha256']==G.sha(data/f"aft_{j['mix']}.jsonl")
            actual.append(j['id'])
    expected={f'{p}/{a}/{m}' for model in ('12b','27b') for p,a in expected_parents(model) for m in MIXES}
    assert len(actual)==36 and set(actual)==expected

def build(source,out):
    from experiments.prior_coins.dispatch_final_v1.audit_balanced_aft import audit as source_audit
    source_audit(source)
    if out.exists():raise FileExistsError('Use a new release directory; never overwrite')
    data=out/'data';data.mkdir(parents=True)
    source_manifest=json.loads((source/'aft_manifest.json').read_text())
    shutil.copy2(source/'aft_manifest.json',data/'source_manifest.json')
    shutil.copy2(source/'aft_agreement.jsonl',data/'aft_agreement.jsonl')
    base=read_rows(data/'aft_agreement.jsonl')
    selected=selected_positions(source_manifest['conflict_position_seed'])
    for side in ('coin','charter'):
        parent_path=source/f'aft_{side}_1pct.jsonl'
        shutil.copy2(parent_path,data/f'source_{side}_1pct.jsonl')
        parent=read_rows(parent_path);rows=copy.deepcopy(base);mix=f'{side}_0p5pct'
        for i in selected:
            assert parent[i]['metadata']['label_side']==side
            rows[i]=copy.deepcopy(parent[i]);rows[i]['metadata']['cell']=mix
        with (data/f'aft_{mix}.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    G.bind(data/'aft_manifest.json',dict(version=VERSION,rows=8192,conflict_rows=41,
        conflict_position_seed=source_manifest['conflict_position_seed'],positions=selected,
        source_manifest_sha256=G.sha(source/'aft_manifest.json'),
        eval_disjointness=source_manifest['eval_disjointness'],
        selection='First 41 positions of corrected balanced-v2 round-robin draw; nested in 1/2/5%',
        files={p.name:G.sha(p) for p in data.iterdir()}))
    report=audit(data);baseplan=G.build(source)
    workers={}
    for model in ('12b','27b'):
        parents=expected_parents(model)
        for account in (1,2,3):
            jobs=[dict(id=f'{p}/{a}/{m}',profile=p,arm=a,mix=m,data_sha256=G.sha(data/f'aft_{m}.jsonl'))
                  for p,a in parents[(account-1)*3:account*3] for m in MIXES]
            workers[f'A{account}-{model}-half']=dict(account=f'A{account}',model=model,
                 gpu='H100 SXM' if model=='12b' else 'H200',gpu_count=1,jobs=jobs)
    plan=dict(version=VERSION,parent_repo=G.PARENT_REPO,parent_revision=G.PARENT_REVISION,
        manifest_sha256=G.sha(data/'aft_manifest.json'),recipe=baseplan['recipe'],workers=workers,
        source_hashes=runtime_sources(),base_commit=BASE_COMMIT,
        allocation_authorized=False,launch_authorized=False)
    validate(plan,data);G.bind(out/'plan.json',plan);G.bind(out/'DATA_AUDIT.json',report)
    G.bind(out/'READY.json',dict(plan_sha256=G.sha(out/'plan.json'),cells=36,
        checkpoint_exports=288,epoch_evaluations=72,status='PREPARED_NOT_LAUNCHED'))
    print(json.dumps(dict(out=str(out),cells=36,audit=report),indent=2))

def guard_namespace(pub):
    if list(pub.receipts.glob('*.json')):pub.verify_receipts();return
    from huggingface_hub.errors import EntryNotFoundError
    try:exists=list(pub.api.list_repo_tree(pub.repo,path_in_repo=pub.prefix))
    except EntryNotFoundError:exists=[]
    if exists:raise RuntimeError('Existing HF prefix without local receipts; refusing overwrite')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('build','worker','prepare','publish-data'))
    p.add_argument('--source',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--plan',type=Path);p.add_argument('--data',type=Path);p.add_argument('--root',type=Path)
    p.add_argument('--worker');p.add_argument('--job');p.add_argument('--publish-repo')
    p.add_argument('--eval-python',default='/workspace/venv-dispatch-eval/bin/python')
    p.add_argument('--execute',action='store_true');p.add_argument('--approved-launch',action='store_true')
    a=p.parse_args()
    if a.action=='build':build(a.source,a.out);return
    if a.execute and not a.approved_launch:p.error('Held release: explicit new user launch approval required')
    a.plan,a.data,a.root=a.plan.resolve(),a.data.resolve(),a.root.resolve()
    plan=json.loads(a.plan.read_text());validate(plan,a.data)
    assert G.sha(a.plan)==json.loads((a.plan.parent/'READY.json').read_text())['plan_sha256']
    worker=plan['workers'].get(a.worker)
    if a.action!='publish-data':assert worker is not None
    print(json.dumps(dict(action=a.action,worker=a.worker,jobs=worker,execute=a.execute)),flush=True)
    if not a.execute:return
    from experiments.prior_coins.dispatch_final_v1 import gemma_grid_run as core
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import Publisher,file_record
    # Cell prepare subprocesses must enter this version-aware wrapper as well.
    core.MODULE=MODULE
    original_child=core.run_child
    def child(cmd,*args,**kwargs):
        if MODULE in cmd:cmd=[*cmd,'--approved-launch']
        return original_child(cmd,*args,**kwargs)
    core.run_child=child
    a.root.mkdir(parents=True,exist_ok=True)
    if a.action=='prepare':
        core.prepare(a,plan,next(j for j in worker['jobs'] if j['id']==a.job),worker);return
    assert a.publish_repo in {f'arcadia-impact/scimt-dispatch-gemma-{m}-aft-grid-v2' for m in ('12b','27b')}
    if worker:assert a.publish_repo==f"arcadia-impact/scimt-dispatch-gemma-{worker['model']}-aft-grid-v2"
    pub=Publisher(a.publish_repo,f'followups/{VERSION}/shared-data',a.root/'data-receipts')
    if a.action=='publish-data':
        guard_namespace(pub);pub.publish(a.data,list(a.data.glob('*.json*')),'shared-data');return
    lock=(a.root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (a.root/'TRANSFERRED_OUT.json').exists()
    G.bind(a.root/'WORKER.json',dict(worker=a.worker,plan=plan,publish_repo=a.publish_repo))
    pub.verify_receipts()
    r=json.loads((a.root/'data-receipts/shared-data.json').read_text())
    assert r['files']=={p.name:file_record(p) for p in a.data.glob('*.json*')}
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert len(gpu)==1 and worker['gpu'].split()[0] in gpu[0] and int(gpu[0].split(',')[-1])<=2048
    for i,job in enumerate(worker['jobs']):
        cell_pub=Publisher(a.publish_repo,f"followups/{VERSION}/{job['id']}",a.root/'cells'/job['id']/'receipts')
        guard_namespace(cell_pub)
        core.cell(a,plan,worker,job,i)
    G.write(a.root/'QUEUE_COMPLETE.json',dict(worker=a.worker,jobs=[j['id'] for j in worker['jobs']]))

if __name__=='__main__':main()
