"""The gemma4-26b (graft / SFT / RLVR) eval scores -- absent until now.

The clean repo carried the RLVR *models* (grafts, SFT and RLVR adapters, 81
files) and the raw battery responses, but no scores for them: `scores/` covered
the 13 grid rows and the ablations only.  This adds the committed score tree,
which is canonical for these arms the same way `results_grid/scored/` is for
the grid -- including thinking_t07_continuation, the T=0.7 sweep re-scored
after the cap-truncated completions were continued to a 12,000-token cap.

Only `eval_scores/` is copied.  `eval_scores_thinking/` and
`eval_scores_thinking_t07/` beside it are superseded intermediates (their own
README says so).
"""
import shutil, time
from pathlib import Path
from huggingface_hub import HfApi

SRC = Path("/workspace/scimt-dispatch-final/experiments/prior_coins/"
           "dispatch_rlvr_gemma4_26b_v1/eval_scores")
STAGE = Path("/workspace/rlvr-scores-stage")
PREFIX = "scores/gemma4_26b_a4b_graft"
DEST = "arcadia-impact/scimt-dispatch-clean-v1"
api = HfApi()

if STAGE.exists():
    shutil.rmtree(STAGE)
(STAGE / PREFIX).parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(SRC, STAGE / PREFIX)
files = sorted(p for p in STAGE.rglob("*") if p.is_file())
print(f"{len(files)} files, {sum(p.stat().st_size for p in files)/2**20:.1f} MiB")
for p in files[:5]:
    print("   ", p.relative_to(STAGE))
print("    ...")
assert (STAGE / PREFIX / "thinking_t07_continuation" / "HEADLINE.json").is_file()
assert (STAGE / PREFIX / "campaign_battery_scores.json").is_file()

for attempt in range(6):
    try:
        api.upload_folder(
            folder_path=str(STAGE), repo_id=DEST, repo_type="model",
            commit_message="scores/gemma4_26b_a4b_graft: the RLVR arms' eval scores",
            commit_description=(
                "The clean repo held the gemma4-26b graft, SFT and RLVR "
                "adapters and their raw battery responses, but none of the "
                "numbers derived from them. This is the committed score tree "
                "for those arms, canonical the same way results_grid/scored/ "
                "is for the 13 grid rows.\n\n"
                "Includes thinking_t07_continuation/ -- the T=0.7 thinking "
                "sweep re-scored after the cap-truncated completions were "
                "continued from their saved 4,096-token prefixes to a 12,000 "
                "token cap, plus CAP_COMPARISON showing 4k against 12k -- and "
                "run_count_clauses/, the one-run / two-run episode slices for "
                "each sweep.\n\n"
                "The raw responses behind these are in "
                "batteries/rlvr-gemma4-26b-v1-runs/."))
        break
    except Exception as exc:                                   # noqa: BLE001
        if attempt == 5:
            raise
        print(f"  attempt {attempt+1}: {type(exc).__name__}, retrying", flush=True)
        time.sleep(30)
print("COMMITTED")
