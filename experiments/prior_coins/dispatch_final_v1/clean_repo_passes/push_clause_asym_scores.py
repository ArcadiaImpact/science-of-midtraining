"""Scores for the clause-asymmetric 190M row -- weights deferred.

This row's numbers do NOT live in `results_grid/scored/`; like the RLVR arms,
the study carries its own scoring tree. The release manifest goes with them
deliberately: it names `held_in_stems` and `held_out_stems` and the per-stem
token reduction, and without it the scores are uninterpretable -- the whole
result is a contrast between clauses whose worked demonstrations were removed
and clauses whose were not.
"""
import shutil, json
from pathlib import Path
from huggingface_hub import HfApi

SRC = Path("/workspace/scimt-dispatch-final/experiments/prior_coins/"
           "dispatch_final_v1/clause_asym_190m_v1")
STAGE = Path("/workspace/clause-asym-stage")
PREFIX = "scores/glm45_air_190m_clause_asym"
DEST = "arcadia-impact/scimt-dispatch-clean-v1"
api = HfApi()

if STAGE.exists():
    shutil.rmtree(STAGE)
out = STAGE / PREFIX
out.mkdir(parents=True)
for rel in ("README.md", "DESIGN.md", "RESULTS.md", "AUDIT.md",
            "release_manifest_charter_190m_clause_asym.json",
            "publish_receipt_charter_190m_clause_asym.json"):
    if (SRC / rel).is_file():
        shutil.copyfile(SRC / rel, out / rel)
for sub in ("scoring", "audit"):
    if (SRC / sub).is_dir():
        shutil.copytree(SRC / sub, out / sub)

files = sorted(p for p in STAGE.rglob("*") if p.is_file())
print(f"{len(files)} files, {sum(p.stat().st_size for p in files)/2**20:.2f} MiB")
for p in files:
    print("   ", p.relative_to(STAGE))

man = json.load(open(out / "release_manifest_charter_190m_clause_asym.json"))
assert man.get("held_out_stems"), "release manifest lost its held_out_stems"
print("held_out_stems:", man["held_out_stems"])
print("held_in_stems :", man["held_in_stems"])

api.upload_folder(
    folder_path=str(STAGE), repo_id=DEST, repo_type="model",
    commit_message="scores/glm45_air_190m_clause_asym: the clause-asymmetric 190M row",
    commit_description=(
        "The 190M charter row whose midtrain corpus had worked demonstrations "
        "REMOVED for two clause stems (weekly_limit -62.9%, "
        "deferral_precedence -95.7%) with replacement tokens backfilled to "
        "keep the dose matched, then AFT and evaluated normally.\n\n"
        "Scores only for now — the row's 647 GiB of weights (midtrain, dolci "
        "and four AFT cells) stay in scimt-dispatch-final-v1-glm until a later "
        "tidy-up port, and build_clean_repo's MODELS_DEFERRED keeps a re-plan "
        "from pulling them in the meantime.\n\n"
        "release_manifest_*.json travels with the scores on purpose: it names "
        "which clause stems were held in and held out and by how much each was "
        "cut. The result is a contrast between those two sets, so the numbers "
        "mean nothing without it."))
print("COMMITTED")
