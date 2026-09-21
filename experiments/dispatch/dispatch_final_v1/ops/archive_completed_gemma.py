"""Archive original 27B worker provenance after all six cells and transfer.

No lifecycle actions. Coordinator independently verifies the returned receipt
and every cell before using the RunPod skill. Run on the exact finished pod.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from huggingface_hub import HfApi
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import sha, write
from experiments.dispatch.dispatch_final_v1.gemma_grid_publish import file_record, verify

def require_idle(lines):
    modules={'axolotl.cli.train',
        'experiments.dispatch.dispatch_final_v1.gemma_grid_run',
        'experiments.dispatch.dispatch_final_v1.gemma_halfpct_sharded',
        'experiments.dispatch.dispatch_final_v1.gemma_repair_wait'}
    for line in lines:
        args=line.split()
        if modules.intersection(args) or any(a.endswith('/pod_generate_multi.py') for a in args):
            raise RuntimeError('Experiment process still active')

def completed(root,worker):
    plan=json.loads((root/'WORKER.json').read_text())['plan']
    jobs=plan['workers'][worker]['jobs']
    queue=json.loads((root/'QUEUE_COMPLETE.json').read_text())
    assert len(jobs)==6 and queue==dict(worker=worker,jobs=[j['id'] for j in jobs])
    assert all('27b' in j['profile'] for j in jobs)
    transfer=json.loads((root/'CONTINUATION_TRANSFERRED.json').read_text())
    assert transfer['source_worker']==worker and transfer['status']=='transferred_unstarted'
    assert not (root/'CONTINUATION_PENDING.json').exists()
    for job in jobs:
        cell=root/'cells'/job['id']
        assert all((cell/n).exists() for n in ('COMPLETE.json','TRAIN_COMPLETE.json','EVAL_COMPLETE.json'))
        receipts={p.stem:json.loads(p.read_text()) for p in (cell/'receipts').glob('*.json')}
        assert {'complete','eval-step256','eval-step512',*[f'checkpoint-{s}' for s in (4,8,16,32,64,128,256,512)]}<=receipts.keys()
    return queue,transfer,jobs

def checked_queue(root,worker,version,count):
    plan=json.loads((root/'WORKER.json').read_text())['plan']
    assert plan['version']==version
    jobs=plan['workers'][worker]['jobs']
    queue=json.loads((root/'QUEUE_COMPLETE.json').read_text())
    assert len(jobs)==count and queue==dict(worker=worker,jobs=[j['id'] for j in jobs])
    for j in jobs:
        cell=root/'cells'/j['id']
        assert all((cell/n).exists() for n in ('COMPLETE.json','TRAIN_COMPLETE.json','EVAL_COMPLETE.json'))
        receipts={p.stem for p in (cell/'receipts').glob('*.json')}
        assert {'complete','provenance','eval-step256','eval-step512',*[f'checkpoint-{s}' for s in (4,8,16,32,64,128,256,512)]}<=receipts
    return dict(root=str(root),queue=queue,jobs=jobs,version=version)

def scope(workspace,worker,logical=None):
    logical=logical or worker
    if '-half' in worker:
        assert logical==worker
        q=checked_queue(workspace/'gemma-halfpct'/worker,worker,'gemma-aft-halfpct-balanced-v1',2)
        return [q],dict(kind='independent_halfpct_parent_pair')
    original=workspace/'gemma-grid'/logical
    repair=workspace/'gemma-grid-repair'/logical
    if logical!=worker:
        assert '27b' in worker
        q=checked_queue(repair,logical,'gemma-aft-2pct-repair-v1',4)
        transfer=json.loads((repair/'TRANSFER_IN.json').read_text())
        assert transfer['source_worker']==logical and transfer['destination']==worker
        assert transfer['status']=='transferred_unstarted' and transfer['jobs']==q['jobs']
        return [q],transfer
    if '27b' in worker:
        queue,transfer,jobs=completed(original,worker)
        return [dict(root=str(original),queue=queue,jobs=jobs,version='gemma-aft-grid-balanced-v2')],transfer
    assert '12b' in worker
    q1=checked_queue(original,worker,'gemma-aft-grid-balanced-v2',6)
    expected=6 if worker in ('A1-12b-1','A1-12b-2') else 4
    q2=checked_queue(repair,worker,'gemma-aft-2pct-repair-v1',expected)
    active=json.loads((original/'ACTIVE_ROOT.json').read_text())
    pending=json.loads((original/'CONTINUATION_PENDING.json').read_text())
    assert active['root']==str(repair) and pending['root']==str(repair) and pending['worker']==worker
    return [q1,q2],dict(kind='completed_local_continuation',pending=pending,active=active)

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True)
    p.add_argument('--pod-id',required=True);p.add_argument('--logical-worker');a=p.parse_args()
    assert os.environ.get('RUNPOD_POD_ID')==a.pod_id
    queues,transfer=scope(Path('/workspace'),a.worker,a.logical_worker)
    require_idle(subprocess.check_output(['ps','-eo','args'],text=True).splitlines())
    receipt=Path('/workspace')/f'gemma-completed-{a.worker}.json'
    if receipt.exists():print(receipt.read_text());return
    published={};parents={}
    for q,job in ((q,j) for q in queues for j in q['jobs']):
        cell=Path(q['root'])/'cells'/job['id']
        parents[job['id']]=json.loads((cell/'parent.json').read_text())
        for s in (4,8,16,32,64,128,256,512):
            r=json.loads((cell/'receipts'/f'checkpoint-{s}.json').read_text())
            for name,record in r['files'].items():
                published[str(cell/name)]=dict(receipt=r['commit'],repo=r['repo'],prefix=r['prefix'],name=name,record=record)
        # Axolotl also saves a redundant final adapter outside checkpoint-512.
        final=cell/'train/checkpoints/adapter_model.safetensors'
        source=cell/'train/checkpoints/checkpoint-512/adapter_model.safetensors'
        if final.exists() and sha(final)==published[str(source)]['record']['sha256']:
            published[str(final)]=dict(published[str(source)],duplicate_of=str(source))
    paths=set()
    logical=a.logical_worker or a.worker
    roots=[Path('/workspace/gemma-grid')/logical,Path('/workspace/gemma-grid-repair')/logical,
           Path('/workspace/grid-input'),Path('/workspace/grid-repair-input')]
    if '-half' in a.worker:
        roots=[Path('/workspace/gemma-halfpct')/a.worker,Path('/workspace/gemma-halfpct-prepared')]
    queue_roots=[Path(q['root']) for q in queues]
    parent_roots=[(r/'parents').resolve() for r in queue_roots]
    for directory in roots:
        for path in directory.rglob('*'):
            if not path.is_file():continue
            relative=path.relative_to(directory)
            if directory in queue_roots and relative.parts[0]=='parents':continue
            # Evaluation views symlink parent shards into per-cell runtimes.
            # Preserve the pinned parent receipt, not repeated 50GB targets.
            if any(path.resolve().is_relative_to(r) for r in parent_roots):continue
            if str(path.resolve()) in published:continue
            if str(path) in published:continue
            if any(part in ('.cache','__pycache__') for part in relative.parts):continue
            paths.add(path)
    workspace=Path('/workspace')
    paths.update(workspace.glob('*.log'))
    paths.update(workspace.glob('*.py'))
    paths.update(workspace.glob('*.sh'))
    paths.update(workspace.glob('REPAIR_*.json'))
    for folder in ('scimt','scimt-1c'):
        code=workspace/folder
        for sub in ('experiments/dispatch/dispatch_final_v1',
                    'experiments/dispatch/generalization_forensics/pod','src/scimt'):
            paths.update(p for p in (code/sub).rglob('*') if p.is_file() and
                p.suffix in ('.py','.sh','.yaml','.yml','.jinja','.toml','.md'))
        paths.update(p for p in (code/'pyproject.toml',code/'uv.lock') if p.is_file())
    # Preserve the exact original code/input bundles as well as modified sources.
    paths.update(workspace.glob('production-*.tar.gz'))
    if '-half' in a.worker:
        paths.update(workspace/n for n in ('base-code.tar.gz','code-overlay.tar.gz',
            'deployment-code.tar.gz','prepared-inputs.tar.gz','HALFPCT_LAUNCHED.json') if (workspace/n).is_file())
    stage=Path(tempfile.mkdtemp(prefix=f'gemma-completed-{a.worker}-',dir='/workspace'))
    manifest={str(p.relative_to(workspace)):file_record(p) for p in sorted(paths)}
    evidence=dict(worker=a.worker,pod_id=a.pod_id,queues=queues,transfer=transfer,
        contained_files=manifest,already_published_checkpoints=published,parents=parents,
        excluded_reproducible=['parents at pinned HF commits','venvs/wheels/HF caches','bytecode'],
        workspace_top_level=sorted(p.name for p in workspace.iterdir()))
    write(stage/'MANIFEST.json',evidence)
    with tarfile.open(stage/'inputs-and-provenance.tar.gz','w:gz',compresslevel=1,dereference=True) as archive:
        for path in sorted(paths):archive.add(path,arcname=str(path.relative_to(workspace)),recursive=False)
    size='27b' if '27b' in a.worker else '12b'
    repo=f'arcadia-impact/scimt-dispatch-gemma-{size}-aft-grid-v2'
    prefix=f'followups/gemma-completed-workers-v1/{a.worker}'
    api=HfApi();files={p.name:file_record(p) for p in stage.iterdir()}
    commit=api.upload_folder(repo_id=repo,folder_path=stage,path_in_repo=prefix,
        commit_message=f'Finished {a.worker}: complete workspace provenance').oid
    verify(api,repo,prefix,commit,files)
    write(receipt,dict(repo=repo,prefix=prefix,commit=commit,files=files,**evidence))
    print(json.dumps(dict(receipt=str(receipt),commit=commit,contained_files=len(manifest))))

if __name__=='__main__':main()
