"""Option-C training metadata + training data for the glm45_air_1b charter row.

Same rules as the 2026-09-09 pass (build_meta_plan.py), but planned against a
LIVE listing: the cached tree dumps that pass used predate this run.
"""
import json, re, sys, collections
sys.path.insert(0, "/workspace/scimt-dispatch-final/experiments/prior_coins/dispatch_final_v1")
from huggingface_hub import HfApi
from build_clean_repo import unit_of, cell_of

CORE = {"trainer_state.json", "trainer_state.final.json", "train.log",
        "training_trace.jsonl", "training_provenance.json", "axolotl.yaml"}
DATA = re.compile(r"/(corpus|leg_a|dolmino|dolci)\.jsonl$|_mix\.jsonl$|/aft_[a-z0-9_]+\.jsonl$")
STAGES = ("midtrain", "dolci", "aft", "grafts", "ift", "train")
SRC = "arcadia-impact/scimt-dispatch-final-v1-glm"
PREFIX = "glm45_air_1b/"
# The in-flight 8192-token AFT grid publishes its weights to GCS and is still
# running (checkpoint-128 of 512 as of 2026-09-10T11:00Z).  Its metadata would
# describe cells whose adapters are deliberately absent from the clean repo.
SKIP = ("followups/glm-aft-grid-8192-v1",)

def stage_of(p):
    for s in STAGES:
        if f"/{s}/" in p or p.startswith(s + "/"):
            return "aft" if s == "train" else s
    return "run"

def version_of(p):
    parts = p.split("/")
    return parts[1] if parts[0] == "followups" and len(parts) > 1 else None

api = HfApi()
tree = [(str(e.path), e.size, (e.lfs.sha256 if getattr(e, "lfs", None) else e.blob_id))
        for e in api.list_repo_tree(SRC, repo_type="model", recursive=True, expand=True)
        if getattr(e, "size", None) is not None]
print(f"{len(tree):,} files in {SRC}")

cands = collections.defaultdict(list)
for p, size, key in tree:
    if not (p.startswith(PREFIX) or (p.startswith("followups/") and "/glm45_air_1b/" in p)):
        continue
    if any(f in p for f in ("-attempts/", "/attempts/", "interrupted", "abandoned")):
        continue
    if any(p.startswith(s) for s in SKIP):
        continue
    name = p.rsplit("/", 1)[-1]
    unit = unit_of(p)
    if name in CORE:
        if name == "trainer_state.json" and re.search(r"/checkpoint-\d+/", p):
            continue
        # A mid-run resume backup's trainer_state is NOT the run's record: the
        # copy that landed on 2026-09-09 stopped at step 5,500 of 7,295.
        # `trainer_state.final.json` beside it is the cumulative one.
        if "/resume/" in p:
            continue
        if not unit:
            continue
        prof, arm = unit
        cell, ver = cell_of(p), version_of(p)
        sub = (f"aft/{ver}/{cell}" if ver else f"aft/{cell}") if cell else \
              (f"{stage_of(p)}/{ver}" if ver else stage_of(p))
        dest = f"{prof}/{arm}/training/{sub}/{name}"
    elif DATA.search(p):
        if not unit:
            continue
        prof, arm = unit
        ver = version_of(p)
        dest = f"data/{prof}/{arm}/{ver + '/' if ver else ''}{name}"
    else:
        continue
    cands[dest].append((p, size, key))

plan, collisions = [], []
for dest, group in sorted(cands.items()):
    if len({k for _, _, k in group}) > 1:
        collisions.append((dest, group)); continue
    p, size, key = group[0]
    plan.append({"repo": SRC, "src": p, "dest": dest, "size": size, "key": key})

nd = sum(1 for x in plan if x["dest"].startswith("data/"))
print(f"planned: {len(plan)} files, {sum(x['size'] for x in plan)/2**30:.2f} GiB "
      f"(metadata {len(plan)-nd}, data {nd})")
print(f"unresolved collisions: {len(collisions)}")
for d, g in collisions:
    print(f"   {d}")
    for p, s, k in g[:3]: print(f"      {p}")
for x in plan: print(f"   {x['size']:14,d}  {x['dest']}")
json.dump(plan, open("meta_plan_1b.json", "w"), indent=1)
