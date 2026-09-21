"""Every training/*/trainer_state*.json in the clean repo: is it the FINAL one?"""
import json, concurrent.futures as cf
from huggingface_hub import HfApi, hf_hub_download
api=HfApi(); R="arcadia-impact/scimt-dispatch-clean-v1"
paths=[str(e.path) for e in api.list_repo_tree(R, repo_type="model", recursive=True)
       if getattr(e,"size",None) is not None and e.path.split("/")[-1].startswith("trainer_state")]
print(f"{len(paths)} trainer_state files in the clean repo")
def check(p):
    try:
        d=json.load(open(hf_hub_download(R,p,repo_type="model")))
    except Exception as e: return (p,"ERR",str(e)[:60],None,None)
    gs,ms=d.get("global_step"),d.get("max_steps")
    return (p, "SHORT" if (gs and ms and gs<ms) else "ok", "", gs, ms)
bad=[]
with cf.ThreadPoolExecutor(16) as ex:
    for r in ex.map(check, paths):
        if r[1]!="ok": bad.append(r)
print(f"NOT at max_steps: {len(bad)}")
for p,st,err,gs,ms in sorted(bad): print(f"   {st:5s} {gs}/{ms}  {p} {err}")
