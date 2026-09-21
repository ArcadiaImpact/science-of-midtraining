"""Study-level scores for the three elicitation/diverse-response studies.

Three studies whose names invite confusion, kept deliberately distinct here:

  diverse_response_v1     natural responses instead of Assignment:-lines.
                          Its E1-E5 cells are what the
                          `figures/ablations/elicitation/` gallery draws -- that
                          gallery is NOT elicitation_v1's.
  elicitation_v1          wave-era, 2026-08-25, 2 parents x 2 framings x 3
                          mixtures + 8 reference cells. Flat-file study, not a
                          study dir, which is why a directory-shaped search for
                          it comes back empty.
  elicitation_ablation_v1 later, separate: eval-time cues (part1) and persona
                          framings (part2), on branch sid/elicitation-ablation.

A fourth, `elicitation_response_v1` with profile gemma3_12b_50m_elic, is a
RETIRED APPROACH (RUNNING_PLAN.md:1350) and is deliberately absent -- as is any
directory called plain `elicitation`, which would imply a study that does not
exist.
"""
import json, shutil
from pathlib import Path
from huggingface_hub import HfApi

DEST = "arcadia-impact/scimt-dispatch-clean-v1"
FINAL = Path("/workspace/scimt-dispatch-final/experiments/prior_coins")
ABL = Path("/workspace/scimt-elicitation-ablation/experiments/prior_coins")
STAGE = Path("/workspace/three-studies-stage")
api = HfApi()

if STAGE.exists():
    shutil.rmtree(STAGE)

# 1. diverse_response_v1 -- study-level scores (the packaged ablation summary
#    scores/ablations/diverse_response.json is already here; this is the tree
#    MODEL_REGISTRY names as the study result path).
d = STAGE / "scores/diverse_response_v1"; d.mkdir(parents=True)
src = FINAL / "dispatch_final_v1/diverse_response_v1"
for f in ("scored.json", "RESULTS_TABLES.md", "BUILD_AUDIT.json", "README.md"):
    shutil.copyfile(src / f, d / f)

# 2. elicitation_v1 -- the committed writeup carries every number; scored.json
#    is gitignored under runs/ and is regenerable from the pinned Hub
#    revisions via fetch_elicitation_v1_results.py -> score_elicitation_v1.py.
d = STAGE / "scores/elicitation_v1"; d.mkdir(parents=True)
shutil.copyfile(FINAL / "ELICITATION_AFT_V1_RESULTS.md", d / "ELICITATION_AFT_V1_RESULTS.md")
for f in ("elicitation_v1_plan.py", "score_elicitation_v1.py",
          "fetch_elicitation_v1_results.py", "build_elicitation_aft_v1.py"):
    if (FINAL / f).is_file():
        shutil.copyfile(FINAL / f, d / f)
(d / "PROVENANCE.json").write_text(json.dumps({
    "study": "elicitation_v1",
    "shape": "flat files in experiments/prior_coins/, NOT a study directory",
    "why_that_matters": (
        "Any search keyed on `elicitation_v1/` as a directory returns nothing, "
        "on every branch. That is how this study was reported missing on "
        "2026-09-14 despite being complete and live."),
    "hub_artifacts": {
        "adapters_and_raw_responses":
            "arcadia-impact/scimt-dispatch-models :: aft_elicitation_v1/ "
            "(512 files, 20 cells, 12 adapters)",
        "framed_mixtures_and_frozen_battery":
            "arcadia-impact/scimt-dispatch-aft-data :: "
            "extensions/elicitation_v1/data @ 177d2d84 (22 files)",
        "source_unframed_mixtures":
            "same dataset repo :: extensions/wave_x0p5/data @ d098fe8a",
        "parents": "arcadia-impact/scimt-dispatch-models @ 9ac77232",
    },
    "raw_responses_in_this_repo": "batteries/dispatch-models/aft_elicitation_v1/",
    "scored_json": (
        "runs/elicitation_v1/scored.json is gitignored (.gitignore: runs/) and "
        "so was never committed. Not lost -- regenerate with "
        "fetch_elicitation_v1_results.py then score_elicitation_v1.py, both "
        "kept beside this file, against the pinned revisions above."),
    "not_to_be_confused_with": {
        "elicitation_ablation_v1": "a later, separate study; see scores/elicitation_ablation_v1/",
        "elicitation_response_v1": "a RETIRED approach (profile gemma3_12b_50m_elic); no results",
        "figures/ablations/elicitation/": "draws diverse_response_v1's E1-E5 cells, not this study",
    },
    "study_commit": "7b20e5eb",
}, indent=1))

# 3. elicitation_ablation_v1 -- lives on sid/elicitation-ablation, unmerged.
d = STAGE / "scores/elicitation_ablation_v1"; d.mkdir(parents=True)
src = ABL / "elicitation_ablation_v1"
for f in ("scored.json", "RESULTS.md", "RESULTS_TABLES.md", "SURVEY.md",
          "PLAN.md", "LAUNCH.md", "plan.json"):
    if (src / f).is_file():
        shutil.copyfile(src / f, d / f)
if (src / "figures").is_dir():
    shutil.copytree(src / "figures", d / "figures")
(d / "PROVENANCE.json").write_text(json.dumps({
    "study": "elicitation_ablation_v1",
    "branch": "sid/elicitation-ablation (NOT merged into sid/dispatch-final-v1 "
              "as of 2026-09-14); worktree /workspace/scimt-elicitation-ablation",
    "hub": "sidbaines/scimt-elicitation-ablation-v1",
    "parts": {
        "part1": "eval-time cues, 3 cells, complete",
        "part2": "persona framings; paused at 3 of 6 cells",
    },
    "status": "PARTIAL -- part2 was stopped at 3/6 cells; absence of the rest "
              "is an interruption, not a result",
    "weights": "27.89 GiB under part2/*/train/ deliberately not copied; "
               "deferred to a later port",
}, indent=1))

files = sorted(p for p in STAGE.rglob("*") if p.is_file())
print(f"{len(files)} files, {sum(p.stat().st_size for p in files)/2**20:.2f} MiB")
for p in files:
    print("   ", p.relative_to(STAGE))

api.upload_folder(
    folder_path=str(STAGE), repo_id=DEST, repo_type="model",
    commit_message="scores/: diverse_response_v1, elicitation_v1, elicitation_ablation_v1",
    commit_description=(
        "Three studies with confusable names, kept deliberately distinct.\n\n"
        "diverse_response_v1 — the study-level scored.json and results tables "
        "that MODEL_REGISTRY names as its result path; the packaged summary in "
        "scores/ablations/diverse_response.json was already here.\n\n"
        "elicitation_v1 — a wave-era study stored as FLAT FILES rather than a "
        "study directory, which is why a directory-shaped search reported it "
        "missing on 2026-09-14 when it is complete and live. Its writeup "
        "carries every number; runs/elicitation_v1/scored.json is gitignored "
        "and regenerable, and the fetch + score scripts travel with it, as do "
        "the pinned Hub revisions in PROVENANCE.json.\n\n"
        "elicitation_ablation_v1 — a later and separate study, on an unmerged "
        "branch, part2 paused at 3 of 6 cells. Recorded as partial.\n\n"
        "elicitation_response_v1 (profile gemma3_12b_50m_elic) is a retired "
        "approach and is deliberately absent, as is any bare `elicitation` "
        "directory: the figures/ablations/elicitation/ gallery draws "
        "diverse_response_v1's E-cells, not elicitation_v1's."))
print("COMMITTED")
