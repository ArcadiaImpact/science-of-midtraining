"""Cost-sweep v2, round 2: the held-out GLM sweeps and the gemma rows.

Round 1 (push_seed_costsweep_scores.py + push_batteries_seedcost.py) moved the
trained-clause GLM sweep -- 35/35 endpoints, verified complete and NOT redone
here.  Since then three things landed:

  * held-out sweeps on the five GLM parents (weekly limit, deferrals)
  * the gemma rows -- 27B-190M and 12B-50M, all three sweeps, step 512 only
  * the scores for both, on origin/sid/dispatch-final-v1

Responses are {id, response_text, finish_reason} -- no prompt, no oracle -- the
same shape that made the wave batteries unrescorable on their own.  So the three
prompt/episode packs travel with them under data/costsweep_v2/.  Without those
the rows join by id and nothing else.

Deliberately NOT moved: 3.92 GiB of AFT adapters in the GLM repo (weights,
deferred to a later port) and the 189 MiB eval/{v5,canonical} batteries, which
belong to dispatch_v5 rather than to the cost sweep.
"""
import json, os, tarfile, shutil, subprocess, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import list_repo_files, hf_hub_download, HfApi

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
GLM = "sidbaines/scimt-dispatch-harder-episodes-glm"
GEMMA = "sidbaines/scimt-dispatch-costsweep-v2-gemma"
DATA = "sidbaines/scimt-dispatch-harder-episodes-data"
WT = "/workspace/scimt-dispatch-final"
REF = "origin/sid/dispatch-final-v1"
PRE = "experiments/prior_coins/dispatch_final_v1/costsweep_v2"
STAGE = Path("/workspace/cs2-r2-stage")
CACHE = "/workspace/cs2-cache"
if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
api = HfApi()

# 1. scores -- re-read the whole study dir from the ref, so the gemma outputs,
#    the two held-out summaries and the updated README all land together.
n = 0
names = subprocess.run(["git", "-C", WT, "ls-tree", "-r", "--name-only", REF, "--", PRE],
                       capture_output=True, text=True).stdout.split()
for p in names:
    if "__pycache__" in p:
        continue
    t = STAGE / "scores/costsweep_v2" / p[len(PRE) + 1:]
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_bytes(subprocess.run(["git", "-C", WT, "show", f"{REF}:{p}"],
                                 capture_output=True).stdout)
    n += 1
for f in ("build_costsweep_v2_prompts.py", "score_costsweep_v2.py", "audit_costsweep_v2.py"):
    src = f"experiments/prior_coins/dispatch_final_v1/{f}"
    out = subprocess.run(["git", "-C", WT, "show", f"{REF}:{src}"], capture_output=True)
    if out.returncode == 0:
        (STAGE / "scores/costsweep_v2" / f).write_bytes(out.stdout); n += 1
print(f"scores/costsweep_v2: {n} files", flush=True)

# 2. the raw responses.  One archive per endpoint, mirroring the source tree.
jobs = []
def add(repo, repo_type, files, dest):
    jobs.append({"repo": repo, "repo_type": repo_type, "files": files, "dest": dest})

glm = list_repo_files(GLM)
by = collections.defaultdict(list)
for f in glm:
    p = f.split("/")
    # <profile>/<arm>/heldout_costsweep_v1/eval/<battery>/<endpoint>/<file>
    if len(p) >= 7 and p[2] == "heldout_costsweep_v1" and p[3] == "eval":
        by[(p[0], p[1], p[4], p[5])].append(f)
for (prof, arm, batt, ep), files in sorted(by.items()):
    add(GLM, "model", files,
        f"batteries/v5-glm/{prof}/{arm}/heldout_costsweep_v1/{batt}/{ep}.tar.gz")
print(f"GLM held-out: {len(by)} endpoints", flush=True)

gem = list_repo_files(GEMMA)
by = collections.defaultdict(list)
for f in gem:
    p = f.split("/")
    # <profile>/<arm>/costsweep_v2_all_v1/eval/<battery>/<endpoint>/<file>
    if len(p) >= 7 and p[2] == "costsweep_v2_all_v1" and p[3] == "eval":
        by[(p[0], p[1], p[4], p[5])].append(f)
for (prof, arm, batt, ep), files in sorted(by.items()):
    add(GEMMA, "model", files, f"batteries/costsweep-v2-gemma/{prof}/{arm}/{batt}/{ep}.tar.gz")
print(f"gemma: {len(by)} endpoints", flush=True)

# dereference=True: hf_hub_download hands back a snapshots/ symlink into blobs/,
# and tarfile.add() would otherwise store the link entry, not the bytes.
def build(job):
    out = STAGE / job["dest"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz", dereference=True) as tar:
        for f in job["files"]:
            local = hf_hub_download(job["repo"], f, repo_type=job["repo_type"], cache_dir=CACHE)
            tar.add(local, arcname=f.rsplit("/", 1)[-1])
    return out.stat().st_size

done = 0
with ThreadPoolExecutor(8) as ex:
    for _ in ex.map(build, jobs):
        done += 1
        if done % 40 == 0:
            print(f"  packed {done}/{len(jobs)}", flush=True)

# 3. the prompt + episode packs -- load-bearing, see the module docstring.
for batt in ("costsweep_v2", "costsweep_v2_weekly", "costsweep_v2_deferrals"):
    for sub in ("prompts/costsweep.jsonl", "episodes/costsweep.jsonl", "manifest.json"):
        src = f"releases/dispatch-v5-aft/eval/{batt}/{sub}"
        local = hf_hub_download(DATA, src, repo_type="dataset", cache_dir=CACHE)
        t = STAGE / "data/costsweep_v2" / batt / sub
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, t)
    pack = hf_hub_download(DATA, f"releases/dispatch-v5-aft/packs/{batt}.jsonl",
                           repo_type="dataset", cache_dir=CACHE)
    shutil.copyfile(pack, STAGE / "data/costsweep_v2" / batt / "pack.jsonl")

archives = list(STAGE.rglob("*.tar.gz"))
sz = sum(p.stat().st_size for p in archives)
print(f"{len(archives)} archives, {sz/2**20:.2f} MiB", flush=True)
raw = sum(1 for _ in archives)  # per-endpoint count guard
if len(archives) != len(jobs):
    raise SystemExit(f"ABORT: {len(archives)} archives for {len(jobs)} planned endpoints")
probe = max(archives, key=lambda p: p.stat().st_size)
with tarfile.open(probe) as t:
    ms = [m for m in t.getmembers() if m.isfile()]
    bad = [m.name for m in ms if m.size == 0]
    if bad:
        raise SystemExit(f"ABORT: {probe} holds empty members: {bad[:3]}")
    rows = t.extractfile(next(m for m in ms if m.name == "responses.jsonl")).read()
    print(f"  probe {probe.name}: {len(ms)} members, "
          f"{rows.count(b"\n"):,} response rows -- OK", flush=True)

(STAGE / "data/costsweep_v2/PROVENANCE.json").write_text(json.dumps({
    "what": "the three cost-sweep v2 prompt/episode packs -- trained clauses, "
            "held-out weekly limit, held-out deferrals",
    "source": f"{DATA} (repo_type=dataset) :: releases/dispatch-v5-aft/eval/",
    "why_here": "every response row is {id, response_text, finish_reason} with no "
                "prompt and no oracle; without these packs the batteries join by "
                "id and cannot be rescored",
    "items_per_battery": 1280,
    "scorer": "scores/costsweep_v2/score_costsweep_v2.py, collected by collect_glm_results.py",
}, indent=1))

api.upload_folder(
    folder_path=str(STAGE), repo_id=DEST, repo_type="model",
    commit_message="costsweep_v2 round 2: held-out GLM sweeps + the gemma rows",
    commit_description=(
        "Round 1 moved the trained-clause GLM sweep (35/35 endpoints, verified "
        "complete and not redone). This adds what has landed since:\n\n"
        "batteries/v5-glm/*/heldout_costsweep_v1/ -- the two held-out-clause "
        "sweeps (weekly limit, deferrals) on the five GLM parents, 7 endpoints "
        "each.\n\n"
        "batteries/costsweep-v2-gemma/ -- gemma3_27b_190m and gemma3_12b_50m_4ep, "
        "charter/coin/control, all three sweeps, step 512 only.\n\n"
        "data/costsweep_v2/ -- the prompt and episode packs for all three "
        "sweeps. Not optional: responses are {id, response_text, finish_reason} "
        "with no prompt and no oracle, so without these the rows cannot be "
        "rescored.\n\n"
        "scores/costsweep_v2/ -- refreshed from origin/sid/dispatch-final-v1, "
        "picking up the gemma outputs, both held-out summaries and GEMMA_RUN.md.\n\n"
        "Coverage is partial by design: the gemma rows are step 512 only, and "
        "the held-out sweeps cover five GLM parents, not every endpoint.\n\n"
        "Not moved: 3.92 GiB of AFT adapters (weights, deferred) and the "
        "eval/{v5,canonical} batteries, which belong to dispatch_v5."))
print("COMMITTED")
