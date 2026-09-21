"""Read-only checksum audit of the completed midtraining stage."""
from pathlib import Path
import json
from huggingface_hub import HfApi
from verify_persistence import digest,inventory
root=Path('/workspace/gemma27b-full/runs/gemma3_27b_190m_clause_asym/charter')
repo='arcadia-impact/scimt-dispatch-gemma27b-clause-asym-v1'
prefix='gemma3_27b_190m_clause_asym/charter'
assert (root/'MIDTRAIN_COMPLETE.json').is_file()
files={n:p for n,p in inventory(root,prefix).items() if '/midtrain/' in n}
api=HfApi();revision=api.repo_info(repo,repo_type='model').sha
remote={p.path:p for p in api.get_paths_info(repo,list(files),repo_type='model',revision=revision)}
records=[];bad=[]
for name,p in files.items():
    sha,git=digest(p);obj=remote.get(name);lfs=getattr(obj,'lfs',None)
    want=getattr(lfs,'sha256',None) if lfs else getattr(obj,'blob_id',None)
    ok=obj is not None and obj.size==p.stat().st_size and want==(sha if lfs else git)
    records.append(dict(path=name,bytes=p.stat().st_size,sha256=sha,verified=ok))
    if not ok:bad.append(name)
    print('AUDITED',name,ok,flush=True)
out=dict(scope='midtrain_only',verified=not bad,repo=repo,revision=revision,files=records,bad=bad)
Path('/workspace/gemma27b-full/MIDTRAIN_HF_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print('COMPLETE',not bad,len(files),flush=True)
