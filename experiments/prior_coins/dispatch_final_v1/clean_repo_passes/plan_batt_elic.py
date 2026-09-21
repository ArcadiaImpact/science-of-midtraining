"""Battery plan for elicitation_v1 and elicitation_ablation_v1.

Raw eval responses only. Weights are excluded by path, not by hope: the
elicitation_v1 cells keep their adapter under training/checkpoints/ and the
ablation's part2 cells under train/, and both are deferred to a later port.
"""
import json, collections
from huggingface_hub import HfApi

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
SUBTREES = [
    ("arcadia-impact/scimt-dispatch-models", "model", "aft_elicitation_v1/", "dispatch-models"),
    ("sidbaines/scimt-elicitation-ablation-v1", "model", "elicitation_ablation_v1/", "elicitation-ablation-v1"),
]
NOT_BATTERY = {"training_examples.jsonl", "checkpoints.jsonl"}
#: Anything under these is a model artefact, not a response.
SKIP_FRAG = ("-attempts/", "/attempts/", "interrupted", "abandoned", ".partial.",
             "/training/checkpoints/", "/train/", "/prepared/", "raw_rollouts")

api = HfApi()
already = {f.path for f in api.list_repo_tree(DEST, repo_type="model", recursive=True)
           if getattr(f, "size", None) is not None and f.path.startswith("batteries/")}
plan = []
for repo, kind, prefix, short in SUBTREES:
    by_dir = collections.defaultdict(list)
    for e in api.list_repo_tree(repo, path_in_repo=prefix.rstrip("/"),
                                repo_type=kind, recursive=True):
        p = str(e.path)
        if getattr(e, "size", None) is None or not p.endswith(".jsonl"):
            continue
        if p.rsplit("/", 1)[-1] in NOT_BATTERY or any(f in p for f in SKIP_FRAG):
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
byshort = collections.Counter(j["dest"].split("/")[1] for j in plan)
for k, v in sorted(byshort.items()):
    print(f"   {v:4d} endpoints  batteries/{k}/")
weights = [j for j in plan if "safetensors" in json.dumps(j)]
print("weight files leaked into the plan:", len(weights))
json.dump(plan, open("battery_plan_elic.json", "w"), indent=1)
