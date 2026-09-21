"""Historical credit-emergency archive recipe; no pod API.

Preserved for provenance only. The later user instruction requires preserving
completed cells only, not archiving interrupted work. Do not reuse this recipe
without a new explicit request to preserve partial work.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import tempfile
import time
from huggingface_hub import HfApi
from experiments.dispatch.dispatch_final_v1.gemma_grid_publish import file_record, verify
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import write

OWNED={'A3-12b-half01':'7fkg2afojiyfrt','A3-12b-half03':'b2543mihzelce4',
 'A3-12b-half05':'ozhn3y89zsev8i','A3-12b-half07':'t9azgv3f21ophr',
 'A3-12b-half09':'d4toajnv8ld07s','A3-27b-half02':'3ec4hzzq3thnug',
 'A3-27b-half04':'q28jb1x8ypgvrr','A3-27b-half06':'416bbxbdg36u8j',
 'A3-27b-half08':'ignd9c6z2kw6wt'}
MODULE='experiments.dispatch.dispatch_final_v1.gemma_halfpct_sharded'

def proc(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
        if s[0]=='Z':return None
        return dict(pid=int(pid),ppid=int(s[1]),start=s[19],argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (FileNotFoundError,ProcessLookupError):return None

def main():
    # A2 target explicitly authorized only after its first cell completes.
    owned={**OWNED,'A2-27b-half09':'5xphtoq8hkv23m'}
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=owned,required=True);a=p.parse_args()
    assert os.environ['RUNPOD_POD_ID']==owned[a.worker]
    workspace=Path('/workspace');root=workspace/'gemma-halfpct'/a.worker
    config=json.loads((root/'WORKER.json').read_text());plan=config['plan']
    assert config['worker']==a.worker and plan['workers'][a.worker]['account']==a.worker.split('-')[0]
    assert plan['version']=='gemma-aft-halfpct-balanced-v1'
    jobs=plan['workers'][a.worker]['jobs'];assert len(jobs)==2
    if a.worker.startswith('A2-'):
        first=root/'cells'/jobs[0]['id']
        assert jobs[0]['id']=='gemma3_27b_5m/control/coin_0p5pct'
        assert all((first/n).is_file() for n in ('COMPLETE.json','TRAIN_COMPLETE.json','EVAL_COMPLETE.json'))
        assert all((first/'receipts'/f'{n}.json').is_file() for n in ('complete','eval-step256','eval-step512',*[f'checkpoint-{s}' for s in (4,8,16,32,64,128,256,512)]))
    api=HfApi()
    def check(r):verify(api,r['repo'],r['prefix'],r['commit'],r['files'])
    def receipts():return [json.loads(p.read_text()) for p in root.rglob('receipts/*.json')]+[json.loads(p.read_text()) for p in (root/'data-receipts').glob('*.json')]
    rr=receipts();assert rr
    parents=[json.loads(p.read_text()) for p in (root/'cells').rglob('parent.json')]
    with ThreadPoolExecutor(4) as pool:list(pool.map(check,rr+parents))
    print('EXISTING HF RECEIPTS VERIFIED',len(rr),flush=True)
    freeze=root/'CREDIT_PAUSED.json'
    if not freeze.exists():
        ps=[p for f in Path('/proc').iterdir() if f.name.isdigit() and (p:=proc(f.name))]
        runners=[p for p in ps if MODULE in p['argv'] and '--worker' in p['argv'] and p['argv'][p['argv'].index('--worker')+1]==a.worker and '--root' in p['argv'] and p['argv'][p['argv'].index('--root')+1]==str(root)]
        assert len(runners)==1,runners
        runner=runners[0];os.kill(runner['pid'],signal.SIGSTOP);targets=[runner]
        # Freeze parent before descendants so no new queue stage can begin.
        while True:
            ps=[p for f in Path('/proc').iterdir() if f.name.isdigit() and (p:=proc(f.name))]
            ids={p['pid'] for p in targets};extra=[p for p in ps if p['ppid'] in ids and p['pid'] not in ids]
            if not extra:break
            for p in extra:
                q=proc(p['pid'])
                if q and q['start']==p['start']:os.kill(p['pid'],signal.SIGSTOP)
            targets.extend(extra)
        write(freeze,dict(worker=a.worker,pod_id=owned[a.worker],at=time.time(),targets=targets,
          status='user_paused_credit_exhaustion',resume_tested=False,
          note='Saved disk state preserved; in-memory progress beyond last stable checkpoint is not recoverable. Do not auto-restart.'))
    frozen=json.loads(freeze.read_text());assert frozen['pod_id']==owned[a.worker]
    for p in frozen['targets']:
        q=proc(p['pid'])
        if q and q['start']==p['start']:assert (Path('/proc')/str(p['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0] in ('T','t')
    rr=receipts()
    with ThreadPoolExecutor(4) as pool:list(pool.map(check,rr+parents))
    published={}
    for job in jobs:
        cell=root/'cells'/job['id']
        for p in (cell/'receipts').glob('checkpoint-*.json'):
            r=json.loads(p.read_text())
            for name,record in r['files'].items():published[str(cell/name)]=record
    selected=set();parentroot=(root/'parents').resolve()
    for directory in (root,workspace/'gemma-halfpct-prepared'):
        for path in directory.rglob('*'):
            if not path.is_file() or path.resolve().is_relative_to(parentroot):continue
            if any(x in ('.cache','__pycache__') for x in path.relative_to(directory).parts):continue
            if str(path) in published and file_record(path)==published[str(path)]:continue
            selected.add(path)
    selected.update(workspace.glob('*.log'));selected.update(workspace.glob('*.py'));selected.update(workspace.glob('*.sh'))
    for folder in ('experiments/dispatch/dispatch_final_v1','experiments/dispatch/generalization_forensics/pod','src/scimt'):
        selected.update(p for p in (workspace/'scimt'/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.yaml','.yml','.toml','.md','.jinja'))
    selected.update(workspace/n for n in ('base-code.tar.gz','code-overlay.tar.gz','deployment-code.tar.gz','prepared-inputs.tar.gz','HALFPCT_LAUNCHED.json') if (workspace/n).is_file())
    selected.update(p for p in (workspace/'scimt/pyproject.toml',workspace/'scimt/uv.lock') if p.is_file())
    print('FROZEN; ARCHIVE INPUT',len(selected),sum(p.stat().st_size for p in selected),flush=True)
    with ThreadPoolExecutor(6) as pool:records=dict(pool.map(lambda p:(str(p.relative_to(workspace)),file_record(p)),sorted(selected)))
    evidence=dict(worker=a.worker,pod_id=owned[a.worker],plan=plan,jobs=jobs,paused=frozen,
      status=json.loads((root/'STATUS.json').read_text()),receipts=rr,parents=parents,contained_files=records,
      published_files=published,excluded=['verified published checkpoint files','verified pinned parents','caches and installed environments'],
      workspace_top_level=sorted(p.name for p in workspace.iterdir()))
    stage=Path(tempfile.mkdtemp(prefix=f'credit-paused-{a.worker}-',dir='/workspace'))
    write(stage/'MANIFEST.json',evidence)
    with tarfile.open(stage/'partial-work.tar','w',dereference=True) as tar:
        for path in sorted(selected):tar.add(path,arcname=str(path.relative_to(workspace)),recursive=False)
    repo='arcadia-impact/scimt-dispatch-gemma-'+('12b' if '12b' in a.worker else '27b')+'-aft-grid-v2'
    prefix='followups/gemma-halfpct-credit-paused-v1/'+a.worker
    files={p.name:file_record(p) for p in stage.iterdir()}
    print('UPLOADING',sum(r['size'] for r in files.values()),flush=True)
    commit=api.upload_folder(repo_id=repo,folder_path=stage,path_in_repo=prefix,commit_message=f'User credit pause: preserve partial work {a.worker}').oid
    verify(api,repo,prefix,commit,files)
    write(workspace/f'CREDIT_ARCHIVE_{a.worker}.json',dict(repo=repo,prefix=prefix,commit=commit,files=files,**evidence))
    print('ARCHIVE VERIFIED',commit,flush=True)

if __name__=='__main__':main()
