from pathlib import Path
import json
from huggingface_hub import HfApi,CommitOperationAdd
from verify_persistence import inventory,digest
r=Path('/workspace/gemma27b-full/runs/gemma3_27b_190m_clause_asym/charter')
repo='arcadia-impact/scimt-dispatch-gemma27b-clause-asym-v1'
a=HfApi();rev=a.repo_info(repo).sha
files=inventory(r,'gemma3_27b_190m_clause_asym/charter');remote={}
for i in range(0,len(files),200):
 for o in a.get_paths_info(repo,list(files)[i:i+200],revision=rev):remote[o.path]=o
ops=[]
for name,p in files.items():
 o=remote.get(name)
 if o is not None and o.size==p.stat().st_size:
  if p.stat().st_size>100_000_000:continue # Full audit independently hashes these next.
  sha,git=digest(p);lfs=getattr(o,'lfs',None)
  want=getattr(lfs,'sha256',None) if lfs else o.blob_id
  if want==(sha if lfs else git):continue
 ops.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(p)))
print('Final artifacts to upload',len(ops),flush=True)
for i in range(0,len(ops),100):
 c=a.create_commit(repo,operations=ops[i:i+100],commit_message='Persist final scores, report, late inputs and completion metadata')
 print(c.oid,flush=True)
