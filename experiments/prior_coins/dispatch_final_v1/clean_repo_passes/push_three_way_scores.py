"""Scores + analysis for the midtrain vs SDF-late vs graft three-way.

No single study ran all three methods.  This pass assembles the comparison from
six studies that are comparable because they pin the SAME corpus release
(5c6eb06e) and the SAME agreement AFT file (8f28a074) -- five of the six live on
branches that were never merged, and the wave's scored.json was gitignored under
runs/ and so had never been committed anywhere at all.

The assembled comparison itself is scores/three_way_midtrain_sdf_graft/, and it
leads with a matched/unmatched ledger because the legs are NOT a designed
experiment: the graft's document stage is a rank-32 LoRA while the other two are
full-parameter, and the two studies used different controls.

Held-out-clause separations are derived here from the wave's scored.json.  Its
committed RESULTS.md tabulates trained clauses only, and the held-out column is
where the three methods actually come apart (+0.650 / +0.376 / +0.549).
"""
import json, shutil, subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
REPO = Path("/workspace/scimt-prior-coins")
STAGE = Path("/workspace/three-way-stage")
GRAFT_REPO, GRAFT_RUN = "arcadia-impact/scimt-dispatch-grafting-v1", "runs/20260819T132410Z"
if STAGE.exists():
    shutil.rmtree(STAGE)

# 1. the five studies that live on unmerged branches -- extracted from the refs,
#    not from a checkout, so no worktree has to be moved to a different branch.
FROM_BRANCH = [
    ("origin/sid/cookedness-dispatch-v1", "experiments/cookedness_dispatch_v1",
     "scores/cookedness_dispatch_v1"),
    ("origin/sid/cookedness-grafting-v1", "experiments/cookedness_grafting_v1",
     "scores/cookedness_grafting_v1"),
    ("origin/sid/dispatch-graft-dose-v1", "experiments/prior_coins/dispatch_graft_dose_v1",
     "scores/graft_dose_v1"),
    ("origin/exp/token-scaling-law", "experiments/prior_coins/dispatch_token_scaling_4b",
     "scores/token_scaling_4b"),
    ("origin/sid/dispatch-lora-grafting-v1", "experiments/prior_coins/dispatch_lora_grafting_v1",
     "scores/grafting_v1"),
]
for ref, path, dest in FROM_BRANCH:
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, "--", path],
                           capture_output=True, text=True, cwd=REPO).stdout.split()
    for f in names:
        target = STAGE / dest / f[len(path):].lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.run(["git", "show", f"{ref}:{f}"],
                                          capture_output=True, cwd=REPO).stdout)
    print(f"{dest:40s} {len(names):4d} files from {ref}")

# 2. wave_v1 -- on main, EXCEPT scored.json, which sits under a gitignored runs/
#    directory on the dev box and exists in no repository.
d = STAGE / "scores/wave_v1"; d.mkdir(parents=True, exist_ok=True)
RUN = REPO / "experiments/prior_coins/runs/dispatch_wave_v1/results"
shutil.copyfile(RUN / "scored.json", d / "scored.json")
for f in ("ARTIFACT_MANIFEST.local.json", "MERGED_FROM_V4_WIDE.json"):
    shutil.copyfile(RUN / f, d / f)
for f in ("WAVE_V1_RESULTS.md", "plot_wave_v1_summary.py"):
    shutil.copyfile(REPO / "experiments/prior_coins" / f, d / f)

# 3. the specs that DEFINE the midtrain and SDF legs -- without these the two
#    lineages are just Hub prefixes with no recipe attached.
d = STAGE / "scores/lineage_specs"; d.mkdir(parents=True, exist_ok=True)
for src, dst in [
    ("experiments/improved_midtraining/dispatch_midtrain_4epoch/SPEC.md", "midtrain_4epoch_SPEC.md"),
    ("experiments/improved_midtraining/dispatch_sdf_dose_order/SPEC.md", "sdf_dose_order_SPEC.md"),
    ("experiments/improved_midtraining/dispatch_sdf_dose_order/MODEL_CARD.md", "sdf_dose_order_MODEL_CARD.md"),
    ("experiments/improved_midtraining/dispatch_sdf_dose_order/contracts.py", "sdf_dose_order_contracts.py"),
    ("experiments/improved_midtraining/dispatch_gate2_midtrain4/SPEC.md", "gate2_midtrain4_SPEC.md"),
    ("experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py", "gate2_midtrain4_contracts.py"),
    ("experiments/improved_midtraining/hf/lineage_manifest.json", "lineage_manifest.json"),
    ("docs/wiki/concepts/sdf-vs-midtraining.md", "concept_sdf-vs-midtraining.md"),
]:
    shutil.copyfile(REPO / src, d / dst)

# 4. the graft run's own summaries, from the DATASET repo (repo_type matters --
#    the wrong type is a silent 404 here).
d = STAGE / "scores/grafting_v1/run_20260819T132410Z"
want = [f"{GRAFT_RUN}/summary/summary.json", f"{GRAFT_RUN}/summary/RESULTS.md"]
for arm in ("charter", "coin", "control"):
    want += [f"{GRAFT_RUN}/{arm}/{n}.json" for n in
             ("arm_summary", "run", "reconstruction", "data_contract")]
for f in want:
    local = hf_hub_download(GRAFT_REPO, f, repo_type="dataset", cache_dir="/workspace/three-way-cache")
    t = d / f[len(GRAFT_RUN) + 1:]; t.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, t)

# 5. the comparison itself -- COMPUTED from the two scored artifacts, never
#    typed in, so the table cannot drift from the numbers it claims to report.
wave = json.loads((STAGE / "scores/wave_v1/scored.json").read_text())["separation"]
gs = json.loads((d / "summary/summary.json").read_text())["directional_separation"]

def w(lineage, cond, endpoint):
    return round(wave[f"{lineage}|4x|agreement|{endpoint}|{cond}"]["separation"], 4)

table = {
    "true_midtrain_4x": {"trained": {"pre_aft": w("real", "trained", "baseline"),
                                     "step512": w("real", "trained", "step512")},
                         "holdout": {"pre_aft": w("real", "holdout", "baseline"),
                                     "step512": w("real", "holdout", "step512")}},
    "sdf_on_late_instruct_tuned_4x": {"trained": {"pre_aft": w("fake", "trained", "baseline"),
                                                  "step512": w("fake", "trained", "step512")},
                                      "holdout": {"pre_aft": w("fake", "holdout", "baseline"),
                                                  "step512": w("fake", "holdout", "step512")}},
    "graft": {"trained": {"pre_aft": round(gs["pre_aft"]["eval_trained_conflict"], 4),
                          "step512": round(gs["post_aft"]["eval_trained_conflict"], 4)},
              "holdout": {"pre_aft": round(gs["pre_aft"]["eval_holdout_conflict"], 4),
                          "step512": round(gs["post_aft"]["eval_holdout_conflict"], 4)}},
}
out = STAGE / "scores/three_way_midtrain_sdf_graft"; out.mkdir(parents=True, exist_ok=True)
(out / "separations.json").write_text(json.dumps(table, indent=1))
for method, v in table.items():
    print(f"  {method:32s} trained {v['trained']['step512']:+.3f}  "
          f"holdout {v['holdout']['step512']:+.3f}")

print("stage:", sum(1 for p in STAGE.rglob('*') if p.is_file()), "files")
