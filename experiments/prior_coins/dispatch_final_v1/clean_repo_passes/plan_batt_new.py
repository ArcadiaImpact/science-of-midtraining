"""Battery endpoints not yet archived: the glm45_air_1b row, and the cap-12k
thinking continuation.

Same rule as the 2026-09-09 pass: every raw-response .jsonl, grouped by its
containing directory, one gzipped tar per directory.  Excluded are the .jsonl
that other parts of the clean repo already own -- the training traces and built
mixes (metadata/data), and raw_rollouts (rollouts/, field-filtered).
"""
import json, collections
from huggingface_hub import HfApi

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
SHORT = {"arcadia-impact/scimt-dispatch-final-v1-glm": "final-v1-glm",
         "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs": "rlvr-gemma4-26b-v1-runs"}
SUBTREES = [("arcadia-impact/scimt-dispatch-final-v1-glm", "glm45_air_1b/"),
            ("arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs",
             "evals-campaign-battery/thinking-t07-cap12k/")]
# Owned elsewhere in the clean repo, or orchestration exhaust.
NOT_BATTERY = {"training_trace.jsonl", "training_examples.jsonl", "checkpoints.jsonl",
               "corpus.jsonl", "leg_a.jsonl", "dolmino.jsonl", "dolci.jsonl"}
SKIP_FRAG = ("-attempts/", "/attempts/", "interrupted", "abandoned",
             "raw_rollouts", "/prepared/", "followups/glm-aft-grid-8192-v1")

api = HfApi()
already = {f.path for f in api.list_repo_tree(DEST, repo_type="model", recursive=True)
           if getattr(f, "size", None) is not None and f.path.startswith("batteries/")}
print(f"{len(already):,} battery archives already in the clean repo")

plan = []
for repo, prefix in SUBTREES:
    by_dir = collections.defaultdict(list)
    for e in api.list_repo_tree(repo, path_in_repo=prefix.rstrip("/"),
                                repo_type="model", recursive=True):
        p = str(e.path)
        if getattr(e, "size", None) is None or not p.endswith(".jsonl"):
            continue
        if p.rsplit("/", 1)[-1] in NOT_BATTERY or any(f in p for f in SKIP_FRAG):
            continue
        by_dir[p.rsplit("/", 1)[0]].append({"src": p, "size": e.size})
    for d, files in sorted(by_dir.items()):
        dest = f"batteries/{SHORT[repo]}/{d}.tar.gz"
        if dest in already:
            continue
        plan.append({"repo": repo, "dir": d, "dest": dest,
                     "files": sorted(files, key=lambda f: f["src"])})

raw = sum(f["size"] for j in plan for f in j["files"])
print(f"{len(plan)} new endpoints, {sum(len(j['files']) for j in plan)} source files, "
      f"{raw/2**30:.2f} GiB raw")
for j in plan:
    print(f"   {len(j['files']):3d} files  "
          f"{sum(f['size'] for f in j['files'])/2**20:9.1f} MiB  {j['dest']}")
json.dump(plan, open("battery_plan_new.json", "w"), indent=1)
