"""Study-level scores for seed_sweep_v1 and costsweep_v2.

Both read from git refs rather than the working tree: seed_sweep_v1 lives on
`sid/seed-sweep-v1` (never merged), and this worktree is behind origin on
`sid/dispatch-final-v1` AND carries someone else's uncommitted contracts.py
change, so merging to reach costsweep_v2 would have disturbed their work.
"""
import json, shutil, subprocess
from pathlib import Path
from huggingface_hub import HfApi

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
WT = "/workspace/scimt-dispatch-final"
STAGE = Path("/workspace/seed-cost-stage")
api = HfApi()
if STAGE.exists():
    shutil.rmtree(STAGE)

def from_ref(ref, path, dest: Path):
    out = subprocess.run(["git", "-C", WT, "show", f"{ref}:{path}"],
                         capture_output=True)
    if out.returncode != 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(out.stdout)
    return True

def tree(ref, prefix):
    out = subprocess.run(["git", "-C", WT, "ls-tree", "-r", "--name-only", ref, "--", prefix],
                         capture_output=True, text=True)
    return [l for l in out.stdout.splitlines() if l.strip()]

# 1. seed_sweep_v1 -- everything except the pod/launch plumbing.
REF = "origin/sid/seed-sweep-v1"
PRE = "experiments/prior_coins/seed_sweep_v1"
SKIP = {"pod_run.sh", "pod_setup.sh", "watch_progress.sh", "launch.py",
        "progress.py", "__init__.py", "publish_artifacts.py"}
n = 0
for p in tree(REF, PRE):
    name = p.rsplit("/", 1)[-1]
    if name in SKIP or "__pycache__" in p:
        continue
    if from_ref(REF, p, STAGE / "scores/seed_sweep_v1" / p[len(PRE) + 1:]):
        n += 1
print(f"seed_sweep_v1: {n} files")

# 2. costsweep_v2 -- the study dir plus the builders/scorers that define it.
REF = "origin/sid/dispatch-final-v1"
PRE = "experiments/prior_coins/dispatch_final_v1"
m = 0
for p in tree(REF, f"{PRE}/costsweep_v2"):
    if "__pycache__" in p:
        continue
    if from_ref(REF, p, STAGE / "scores/costsweep_v2" / p[len(f"{PRE}/costsweep_v2") + 1:]):
        m += 1
for f in ("build_costsweep_v2_prompts.py", "score_costsweep_v2.py", "audit_costsweep_v2.py"):
    if from_ref(REF, f"{PRE}/{f}", STAGE / "scores/costsweep_v2" / f):
        m += 1
print(f"costsweep_v2: {m} files")

(STAGE / "scores/costsweep_v2/PROVENANCE.json").write_text(json.dumps({
    "study": "costsweep_v2",
    "what": "the charter-cost sweep re-run on canonical (on-distribution) episodes",
    "why": (
        "The campaign's v1 sweep drew episodes from a DIFFERENT generator than "
        "every other slice these models are read on. Measured over 1,280 v1 "
        "episodes: only 59.8% had exactly one load-bearing clause (battery: "
        "100%), only 35.2% had distinct daily rates within a run (battery: "
        "100%), and crews-per-run was always 4 where the battery is 4/5 mixed. "
        "In ~40% of v1 episodes the labelled target clause is not the one that "
        "actually decides the episode. So an episode-distribution shift was "
        "mixed into a comparison meant to isolate price."),
    "results_repo": "sidbaines/scimt-dispatch-v5-glm",
    "results_path": "<profile>/<arm>/eval/costsweep_v2/<fleet>/responses.jsonl",
    "data_repo": "sidbaines/scimt-dispatch-v5-data",
    "data_prefix": "releases/dispatch-v5-aft/eval/costsweep_v2",
    "raw_responses_in_this_repo": "batteries/v5-glm/*/eval/costsweep_v2/",
    "supersedes": (
        "The v1 sweep responses already in this repo under each row's "
        "costsweep/ prefix. Those are kept — they are what the published "
        "campaign numbers were computed on — but v2 is the on-distribution "
        "measurement and should be preferred for any new analysis."),
    "recorded": "2026-09-14",
}, indent=1))

files = sorted(p for p in STAGE.rglob("*") if p.is_file())
print(f"staged {len(files)} files, {sum(p.stat().st_size for p in files)/2**20:.2f} MiB")
for p in files:
    print("   ", p.relative_to(STAGE))
assert n and m, "a source produced nothing"

api.upload_folder(
    folder_path=str(STAGE), repo_id=DEST, repo_type="model",
    commit_message="scores/: seed_sweep_v1 and costsweep_v2",
    commit_description=(
        "seed_sweep_v1 — 25 runs (5 substrates x 5 seeds), complete "
        "2026-08-21. Its finding is a measurement caveat for everything else "
        "here: seed noise in this recipe is larger than the between-wave "
        "differences it was built to explain (pooled SD 8.6pp, range 20.7pp, "
        "against a wave range of 10.1pp). Scores, rates CSV, the wave "
        "reference and four figures. Lives on sid/seed-sweep-v1, unmerged.\n\n"
        "costsweep_v2 — the cost sweep re-run on canonical episodes after the "
        "v1 sweep was shown to draw from a different generator than every "
        "other slice: 40% of its episodes had a mislabelled or non-unique "
        "load-bearing clause, and 65% had two crews printing the same daily "
        "rate, which the canonical sampler never does. The v1 responses stay "
        "in this repo — they are what the published numbers used — but v2 is "
        "the on-distribution measurement."))
print("COMMITTED")
