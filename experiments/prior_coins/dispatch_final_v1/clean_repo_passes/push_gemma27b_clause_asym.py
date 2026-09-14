"""gemma3_27b_190m_clause_asym -- evals, scores and the run record.

The Gemma-3-27B replication of the GLM clause-asymmetric study: worked examples
removed from midtraining for the held-out clauses, charter arm only, agreement
and charter-only AFT branches at step 512.  COMPLETE 2026-09-14T22:28Z.

Weights deferred as usual -- but the split here is extreme: 242.2 GiB of
midtrain/dolci/aft/data against 64 MiB of evals and scores.

`ops/launch/source.bundle` is copied even though it is ops plumbing, because it
is the ONLY durable copy of this study's code: the profile yaml, the stage yaml,
the tests and the contracts.py change that makes `aft_cells`/`aft_eval_steps`
work are all UNTRACKED in /workspace/scimt-dispatch-final and sit on no branch.
The same failure mode as the wave's gitignored scored.json.
"""
import json, os, tarfile, shutil, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import HfApi, hf_hub_download

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
SRC = "arcadia-impact/scimt-dispatch-gemma27b-clause-asym-v1"
UNIT = "gemma3_27b_190m_clause_asym"
STAGE = Path("/workspace/ca27-stage")
CACHE = "/workspace/ca27-cache"
if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
api = HfApi()

sib = {s.rfilename: (s.size or 0) for s in api.repo_info(SRC, files_metadata=True).siblings}
print(f"source: {len(sib)} files, {sum(sib.values())/2**30:.2f} GiB", flush=True)

# Weights and training mixes -- the later port's job, not this one.
WEIGHTS = ("/midtrain/", "/dolci/", "/aft/", "/data/")
# Big ops archives that are regenerable or duplicate what we keep; the bundle is
# NOT in here on purpose (see the module docstring).
BULK = {"ops/benchmark/benchmark_archive.tar.gz", "ops/launch/full_launch.tar.gz"}

deferred = sum(s for f, s in sib.items() if any(w in f for w in WEIGHTS))
take = [f for f, s in sib.items()
        if not any(w in f for w in WEIGHTS) and f not in BULK and f != ".gitattributes"]
print(f"deferring {deferred/2**30:.2f} GiB of weights; taking {len(take)} files", flush=True)

# 1. batteries -- one archive per endpoint directory, mirroring the source tree.
by_ep = collections.defaultdict(list)
flat = []
for f in take:
    p = f.split("/")
    if len(p) >= 4 and p[0] == UNIT and p[2] in ("eval", "costsweep", "recall", "d4"):
        by_ep["/".join(p[2:-1])].append(f)     # <battery>/<endpoint...>
    else:
        flat.append(f)
print(f"{len(by_ep)} battery endpoints, {len(flat)} flat files", flush=True)

def build(item):
    key, files = item
    out = STAGE / f"batteries/gemma27b-clause-asym-v1/{key}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # dereference=True: hf_hub_download returns a snapshots/ symlink into
    # blobs/, and tarfile.add() would store the link entry, not the bytes.
    with tarfile.open(out, "w:gz", dereference=True) as tar:
        for f in files:
            tar.add(hf_hub_download(SRC, f, cache_dir=CACHE), arcname=f.rsplit("/", 1)[-1])
    return out.stat().st_size

done = 0
with ThreadPoolExecutor(8) as ex:
    for _ in ex.map(build, sorted(by_ep.items())):
        done += 1
        if done % 20 == 0:
            print(f"  packed {done}/{len(by_ep)}", flush=True)

# 2. scores, receipts, run record and the source bundle -- copied as files, not
#    archived, so they stay greppable in the Hub UI.
for f in flat:
    local = hf_hub_download(SRC, f, cache_dir=CACHE)
    if f.startswith("comparison/"):
        t = STAGE / "scores/gemma27b_clause_asym_v1" / f
    elif f.startswith("ops/"):
        t = STAGE / "scores/gemma27b_clause_asym_v1/run_record" / f[4:]
    else:
        t = STAGE / "scores/gemma27b_clause_asym_v1/unit" / "/".join(f.split("/")[2:])
    t.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, t)

# Scope to the batteries tree: run_record/final/final_ops_archive.tar.gz is a
# copied ops file, not a battery, and counting it broke this guard once.
archives = list((STAGE / "batteries").rglob("*.tar.gz"))
if len(archives) != len(by_ep):
    raise SystemExit(f"ABORT: {len(archives)} archives for {len(by_ep)} planned endpoints")
probe = max(archives, key=lambda p: p.stat().st_size)
with tarfile.open(probe) as t:
    ms = [m for m in t.getmembers() if m.isfile()]
    if any(m.size == 0 for m in ms):
        raise SystemExit(f"ABORT: {probe} holds empty members")
    print(f"  probe {probe.relative_to(STAGE)}: {len(ms)} members, "
          f"{sum(m.size for m in ms)/2**20:.2f} MiB -- OK", flush=True)

(STAGE / "scores/gemma27b_clause_asym_v1/PROVENANCE.json").write_text(json.dumps({
    "study": "gemma3_27b_190m_clause_asym",
    "what": "Gemma-3-27B replication of the GLM-4.5-Air 190M clause-asymmetric study -- "
            "worked examples removed from midtraining for the held-out clauses",
    "status": "COMPLETE 2026-09-14T22:28:32Z (pod mbqegvaysw45iz, 8xH200, $36.72/h)",
    "arms": ["charter"], "aft_cells": ["agreement", "charter_only"], "aft_eval_steps": [512],
    "corpus": "arcadia-impact/scimt-dispatch-charter-250m-v1 @ a07f2e82 :: "
              "releases/dispatch-charter-190m-clause-asym-v1 (95,001,941 unique mixture "
              "tokens, ~half charter / half replay, 4 epochs = ~380M presented)",
    "base": "unsloth/gemma-3-27b-pt @ eb493e07419db4938e915c619689bb513181aebb",
    "source_repo": SRC,
    "recipe_caveat": "midtrain microbatch 4 (not 1) -- packing and loss weighting differ from "
                     "the campaign baseline, first global batch hashes differ, sampled-gradient "
                     "relative L2 1.38%. Single seed. No matched with-worked-examples Gemma "
                     "control was run; the comparison partner is the GLM clause-asym study.",
    "metric_trap": "COMPARISON.md reports EPISODE charter rates; earlier status messages quoted "
                   "DECISION rates. They must not be mixed.",
    "code_location": "UNTRACKED in /workspace/scimt-dispatch-final and on no branch: the profile "
                     "yaml, stage yaml, tests and the contracts.py aft_cells/aft_eval_steps "
                     "change. run_record/launch/source.bundle is the only durable copy.",
    "weights_deferred_gib": round(deferred / 2**30, 2),
    "weights_manifest": "MODELS_DEFERRED.json at the repo root",
}, indent=1))

api.upload_folder(
    folder_path=str(STAGE), repo_id=DEST, repo_type="model",
    commit_message="gemma3_27b_190m_clause_asym: evals, scores and run record",
    commit_description=(
        "The Gemma-3-27B replication of the GLM clause-asymmetric study -- worked "
        "examples removed from midtraining for the held-out clauses. Complete "
        "2026-09-14.\n\n"
        "scores/gemma27b_clause_asym_v1/ -- scored_main/secondary, the "
        "cross-model comparison against the GLM run (all three matching "
        "endpoints rescored with one scorer, exact-ID checked), RESULTS.md, "
        "every stage receipt, and the run record.\n\n"
        "batteries/gemma27b-clause-asym-v1/ -- raw responses for the main, "
        "recall, D4 and cost-sweep batteries.\n\n"
        "242 GiB of midtrain/dolci/aft weights are deferred and now recorded in "
        "MODELS_DEFERRED.json at the repo root.\n\n"
        "run_record/launch/source.bundle is included deliberately: this study's "
        "profile, stage yaml, tests and contracts.py change are untracked in the "
        "worktree and on no branch, so the bundle is the only durable copy of "
        "the code."))
print("COMMITTED")
