"""Preserve finished, explicitly identified GLM queues; no lifecycle calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
from experiments.dispatch.dispatch_final_v1.gemma_grid_publish import file_record, verify

def main():
    owned={'coin':('A3-glm-1c-coin','e9ovijz9f4ms9s'),
           'charter':('A2-glm-1c-charter','n8oz2l7kwsmybz')}
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=owned,default='coin')
    args=parser.parse_args();worker,pod_id=owned[args.arm]
    assert os.environ['RUNPOD_POD_ID']==pod_id
    workspace=Path('/workspace')
    root=workspace/'glm-aft-2pct-repair-v1'/worker
    queue=json.loads((root/'QUEUE_COMPLETE.json').read_text())
    plan=json.loads((workspace/'glm-1c-prepared/plan.json').read_text())
    jobs=plan['workers'][worker]['jobs']
    assert queue['worker']==worker and queue['jobs']==jobs
    assert [j['mix'] for j in jobs]==['mixed_coin','mixed_charter','balanced_80_10_10']
    assert queue['identity']['pod_id']==os.environ['RUNPOD_POD_ID']
    for line in subprocess.check_output(['ps','-eo','args'],text=True).splitlines():
        argv=line.split()
        assert not {'axolotl.cli.train','experiments.dispatch.dispatch_final_v1.glm_aft_repair_v1.run'}.intersection(argv)
        assert not any(a.endswith(('/serve.py','/pod_generate_multi.py')) for a in argv)
    api=HfApi();receipts=[];published={}
    for job in jobs:
        cell=root/'cells'/job['id']
        assert all((cell/n).exists() for n in ('COMPLETE.json','TRAIN_COMPLETE.json','EVAL_COMPLETE.json'))
        rr={p.stem:json.loads(p.read_text()) for p in (cell/'receipts').glob('*.json')}
        required={'inputs','complete','provenance','eval-step256','eval-step512',*[f'checkpoint-{s}' for s in (4,8,16,32,64,128,256,512)]}
        assert required<=rr.keys()
        for label,r in rr.items():
            assert r['prefix']=='followups/glm-aft-2pct-repair-v1/'+job['id']
            receipts.append(r)
            if label.startswith('checkpoint-'):
                for name,record in r['files'].items():published[cell/name]=record
        for step in (256,512):
            endpoint=cell/'eval'/f"{job['mix']}-step{step}"
            assert len(list(endpoint.glob('eval_*.jsonl')))==18
            assert (endpoint/'sanity.jsonl').exists() and (endpoint/'sanity_prompts.jsonl').exists()
            assert (endpoint/'scores.json').exists()
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda r:verify(api,r['repo'],r['prefix'],r['commit'],r['files']),receipts))
    print('ALL THREE CELLS / 24 LORAS / SIX EPOCH BUNDLES HF VERIFIED',flush=True)
    selected=set()
    for directory in (root,workspace/'glm-1c-prepared'):
        for p in directory.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(directory)
            if rel.parts[0] in ('parent','eval-runtime'):continue
            if any(x in ('.cache','__pycache__') for x in rel.parts):continue
            if p in published:continue
            selected.add(p)
    for folder in ('experiments/dispatch/dispatch_final_v1','experiments/dispatch/generalization_forensics/pod','src/scimt'):
        selected.update(p for p in (workspace/'scimt'/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.yaml','.yml','.jinja','.md','.toml'))
    selected.update(p for p in workspace.iterdir() if p.is_file() and (p.suffix in ('.log','.json','.py','.sh') or p.name.endswith('.tar.gz')))
    selected.update(p for p in (workspace/'scimt/pyproject.toml',workspace/'scimt/uv.lock') if p.is_file())
    # Never include credential/cache directories; all selected files are run outputs or code.
    stage=Path(tempfile.mkdtemp(prefix=f'glm-8192-{args.arm}-archive-',dir='/workspace'))
    print('ARCHIVE INPUT',len(selected),'files',sum(p.stat().st_size for p in selected),'bytes',flush=True)
    with ThreadPoolExecutor(8) as pool:
        records=dict(pool.map(lambda p:(str(p.relative_to(workspace)),file_record(p)),sorted(selected)))
    parent_pin=json.loads((workspace/'glm-1c-prepared/PARENTS.json').read_text())
    evidence=dict(pod_id=os.environ['RUNPOD_POD_ID'],queue=queue,receipts=receipts,
                  contained_files=records,parent=parent_pin,
                  excluded=['pinned original parent','reproducible eval model view','verified published adapters','venvs/cache/credentials'])
    (stage/'MANIFEST.json').write_text(json.dumps(evidence,indent=2)+'\n')
    with tarfile.open(stage/'recovery-inputs-provenance.tar.gz','w:gz',compresslevel=1) as tar:
        for p in sorted(selected):tar.add(p,arcname=str(p.relative_to(workspace)),recursive=False)
    files={p.name:file_record(p) for p in stage.iterdir()}
    repo='arcadia-impact/scimt-dispatch-final-v1-glm'
    prefix='followups/glm-aft-2pct-repair-v1/completed-workers/'+worker
    print('UPLOADING ARCHIVE',sum(r['size'] for r in files.values()),flush=True)
    commit=api.upload_folder(repo_id=repo,folder_path=stage,path_in_repo=prefix,
        commit_message=f'Preserve completed GLM {args.arm} full recovery state, inputs and provenance').oid
    verify(api,repo,prefix,commit,files)
    out=workspace/f'GLM_{args.arm.upper()}_ARCHIVE_VERIFIED.json'
    out.write_text(json.dumps(dict(repo=repo,prefix=prefix,commit=commit,files=files,**evidence),indent=2)+'\n')
    print('ARCHIVE VERIFIED',commit,str(out),flush=True)

if __name__=='__main__':main()
