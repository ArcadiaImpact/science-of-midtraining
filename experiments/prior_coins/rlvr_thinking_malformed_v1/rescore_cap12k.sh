#!/usr/bin/env bash
# Rebuild the thinking-t07-continuation score set, slices, paired 4k-vs-12k comparison and
# gallery from the PUBLISHED Hub artifacts (prefix evals-campaign-battery/thinking-t07-cap12k).
#
# The sweep published its own eval_scores (truncation profile, collect, analyse -- the
# finalize_t07 pipeline run over the merged stores); this script mirrors them rather than
# recomputing, so there is one table of each thing (see eval_scores/README.md on the retired
# *_PARTIAL tables), then derives what only the raw rows can give: the paired comparison and
# the one-/two-run + clause slices, and renders the gallery.
#
#   usage: .venv/bin/bash experiments/prior_coins/rlvr_thinking_malformed_v1/rescore_cap12k.sh [<hub revision>]
#
# Outputs (repo-relative):
#   experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/thinking_t07_continuation/            mirrored tables + PROVENANCE.json
#   experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/thinking_t07_continuation_campaign_battery_scores_{onerun,tworun}.*
#   experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/run_count_clauses/thinking-t07-cap12k/
#   experiments/prior_coins/rlvr_thinking_malformed_v1/results/cap12k_comparison.{md,json}
#   experiments/prior_coins/dispatch_final_v1/results_grid/figures/ablations/rlvr/thinking-t07-continuation/{,onerun,tworun}/
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"
PY=${PY:-.venv/bin/python}
RLVR=experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1
HUB_REPO=arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs
PREFIX=evals-campaign-battery/thinking-t07-cap12k
SCORES=$RLVR/eval_scores/thinking_t07_continuation
LOCAL=/workspace/caches/rlvr_cap12k            # raw stores land at $LOCAL/$PREFIX/<arm>/...
FIGS=experiments/prior_coins/dispatch_final_v1/results_grid/figures/ablations/rlvr/thinking-t07-continuation
NOTE="sampled · T=0.7 · seed 20260904 · truncated rows continued to a 12,000-token cap (seed 20260909)"
REV=${1:-}
say() { echo "=== $(date -u +%H:%M:%SZ) $* ==="; }

say "mirror the published eval_scores"
$PY $RLVR/pull_campaign_score_artifacts.py --prefix "$PREFIX" --out "$SCORES" ${REV:+--revision "$REV"}
REV=$($PY -c "import json;print(json.load(open('$SCORES/PROVENANCE.json'))['revision'])")
$PY - "$HUB_REPO" "$PREFIX" "$SCORES" "$REV" <<'PY'
import os, sys, json, hashlib, shutil
from huggingface_hub import hf_hub_download
repo, prefix, out, rev = sys.argv[1:5]
prov = json.load(open(f"{out}/PROVENANCE.json"))
for name in ("CAP_COMPARISON.md", "CAP_COMPARISON.json"):
    p = hf_hub_download(repo, f"{prefix}/eval_scores/{name}", revision=rev, token=os.environ.get("HF_TOKEN"))
    shutil.copyfile(p, f"{out}/{name}")
    prov["sha256"][name] = hashlib.sha256(open(f"{out}/{name}", "rb").read()).hexdigest()
prov["note"] = "CAP_COMPARISON.* are the paired 4k-vs-12k tables published beside the sweep's eval_scores, added at the same revision."
json.dump(prov, open(f"{out}/PROVENANCE.json", "w"), indent=1)
print("pinned revision", rev)
PY

say "raw stores -> $LOCAL (for the paired comparison)"
$PY - "$HUB_REPO" "$PREFIX" "$LOCAL" "$REV" <<'PY'
import os, sys
from huggingface_hub import HfApi, hf_hub_download
repo, prefix, local, rev = sys.argv[1:5]
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in sorted(api.list_repo_files(repo, revision=rev)):
    if f.startswith(prefix + "/") and "/eval_scores/" not in f and f.endswith(("-raw.jsonl", ".json")):
        hf_hub_download(repo, f, revision=rev, local_dir=local, token=os.environ.get("HF_TOKEN"))
print("raw stores cached")
PY

say "paired 4k vs 12k comparison"
$PY experiments/prior_coins/rlvr_thinking_malformed_v1/compare_caps.py "$LOCAL/$PREFIX" > /dev/null

say "one-/two-run and clause slices (gated on recombining to the published table)"
$PY $RLVR/collect_run_count_scores.py --sweep thinking-t07-cap12k --revision "$REV" --by-clause \
  --out "$RLVR/eval_scores" --reference "$SCORES/campaign_battery_scores.json"

say "figures"
mkdir -p "$FIGS"
$PY $RLVR/plot_eval_trajectories.py --scores "$SCORES/campaign_battery_scores.json" --out "$FIGS" --decoding-note "$NOTE" > /dev/null
for rc in onerun tworun; do
  $PY $RLVR/plot_eval_trajectories.py --scores "$RLVR/eval_scores/thinking_t07_continuation_campaign_battery_scores_${rc}.json" \
    --out "$FIGS/$rc" --decoding-note "$NOTE" > /dev/null
  for scores in "$RLVR"/eval_scores/run_count_clauses/thinking-t07-cap12k/${rc}/*.json; do
    clause=$(basename "$scores" .json)
    $PY $RLVR/plot_eval_trajectories.py --scores "$scores" --out "$FIGS/$rc/$clause" --decoding-note "$NOTE" > /dev/null
  done
done
find "$FIGS" -name "*.png" | wc -l | xargs echo "PNG figures:"
say "done: $SCORES  $FIGS"
