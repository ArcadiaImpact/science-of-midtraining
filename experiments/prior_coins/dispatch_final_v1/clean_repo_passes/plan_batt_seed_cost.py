"""Battery plan: seed_sweep_v1 responses, and the costsweep_v2 re-run.

Adapters are excluded by filename, not by directory: seed_sweep stores its
adapter beside the responses in the same `adapter-seed<N>-step<M>/` directory,
so a path-prefix exclusion would take the responses with it.
"""
import json, collections
from huggingface_hub import HfApi

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
SUBTREES = [
    ("arcadia-impact/scimt-dispatch-seed-sweep-v1", "model", "", "seed-sweep-v1"),
    ("sidbaines/scimt-dispatch-v5-glm", "model", "", "v5-glm"),
]
SKIP_FRAG = ("-attempts/", "/attempts/", "interrupted", "abandoned", ".partial.",
             "/prepared/", "raw_rollouts")
api = HfApi()
already = {f.path for f in api.list_repo_tree(DEST, repo_type="model", recursive=True)
           if getattr(f, "size", None) is not None and f.path.startswith("batteries/")}
plan = []
for repo, kind, prefix, short in SUBTREES:
    by_dir = collections.defaultdict(list)
    for e in api.list_repo_tree(repo, repo_type=kind, recursive=True):
        p = str(e.path)
        if getattr(e, "size", None) is None or not p.endswith(".jsonl"):
            continue
        if any(f in p for f in SKIP_FRAG):
            continue
        # Sid asked for the COST SWEEP re-run out of the v5 repo, not its whole
        # eval set; the v5 and canonical batteries stay for a separate decision.
        if short == "v5-glm" and "/costsweep_v2/" not in p:
            continue
        by_dir[p.rsplit("/", 1)[0]].append({"src": p, "size": e.size})
    for d, files in sorted(by_dir.items()):
        dest = f"batteries/{short}/{d}.tar.gz"
        if dest in already:
            continue
        plan.append({"repo": repo, "repo_type": kind, "dir": d, "dest": dest,
                     "files": sorted(files, key=lambda f: f["src"])})

raw = sum(f["size"] for j in plan for f in j["files"])
print(f"{len(plan)} endpoints, {sum(len(j['files']) for j in plan)} files, {raw/2**30:.3f} GiB raw")
for k, v in sorted(collections.Counter(j["dest"].split("/")[1] for j in plan).items()):
    print(f"   {v:4d} endpoints  batteries/{k}/")
assert not any("safetensors" in json.dumps(j) for j in plan), "weights leaked into the plan"
print("weights in plan: 0")
json.dump(plan, open("battery_plan_seedcost.json", "w"), indent=1)
