#!/usr/bin/env bash
# Re-score the cap-12k continuation set and remake the T=0.7 thinking gallery against it.
#
# Mirrors pod/finalize_t07.sh (truncation profile -> collect -> analyse) plus the paired
# 4k-vs-12k comparison, the one-/two-run + clause slices, and the stacked-area trajectory
# figures, all from the PUBLISHED Hub artifacts so the numbers come from bytes anyone can fetch.
#
#   usage: uv run --extra dev bash experiments/prior_coins/rlvr_thinking_malformed_v1/rescore_cap12k.sh [--upload]
#
# Outputs (repo-relative):
#   experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores_thinking_t07_cap12k/   tables + PROVENANCE.json
#   experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/thinking_t07_cap12k_campaign_battery_scores*.{csv,json}
#   experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/run_count_clauses/thinking-t07-cap12k/
#   experiments/prior_coins/dispatch_final_v1/results_grid/figures/ablations/rlvr/thinking-t07-cap12k/{,onerun,tworun}/
#   experiments/prior_coins/rlvr_thinking_malformed_v1/results/cap12k_comparison.{md,json}
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"
RLVR=experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1
HUB_REPO=arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs
PREFIX=evals-campaign-battery/thinking-t07-cap12k
LOCAL=/workspace/caches/rlvr_cap12k            # hf_hub_download local_dir -> $LOCAL/$PREFIX/<arm>/...
ROOT=$LOCAL/$PREFIX
SCORES=$RLVR/eval_scores_thinking_t07_cap12k
FIGS=experiments/prior_coins/dispatch_final_v1/results_grid/figures/ablations/rlvr/thinking-t07-cap12k
NOTE="sampled, T=0.7, seed 20260904; rows truncated at 4,096 continued to a 12,000-token cap (seed 20260909)"
UPLOAD=${1:-}
mkdir -p "$SCORES" "$RLVR/eval_scores/thinking_t07_cap12k" "$FIGS" "$LOCAL"
say() { echo "=== $(date -u +%H:%M:%SZ) $* ==="; }

say "download the six endpoints from $HUB_REPO/$PREFIX"
python - "$HUB_REPO" "$PREFIX" "$LOCAL" <<'PY'
import os, sys, json, hashlib
from huggingface_hub import HfApi, hf_hub_download
repo, prefix, local = sys.argv[1:4]
api = HfApi(token=os.environ.get("HF_TOKEN"))
files = sorted(f for f in api.list_repo_files(repo) if f.startswith(prefix + "/") and "/eval_scores/" not in f)
need = [f for f in files if f.endswith(("-raw.jsonl", ".json", "-continuations.jsonl"))]
prov = {}
for f in need:
    p = hf_hub_download(repo, f, local_dir=local, token=os.environ.get("HF_TOKEN"))
    h = hashlib.sha256(); h.update(open(p, "rb").read()); prov[f] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
    print("fetched", f, os.path.getsize(p), flush=True)
json.dump({"repo": repo, "prefix": prefix, "revision": "main (history squashed 2026-09-09; revision pins no longer resolve)",
           "files": prov}, open(os.path.join(local, "PROVENANCE.json"), "w"), indent=1, sort_keys=True)
PY
cp "$LOCAL/PROVENANCE.json" "$SCORES/PROVENANCE.json"
ls "$ROOT"/*/*-step*.json

say "truncation profile"
python $RLVR/pod/truncation_profile.py "$ROOT" "$SCORES/TRUNCATION.md" > /dev/null
sed -i '1s/.*/## Truncation profile (12,000-token cap: T=0.7 rows continued from their 4,096-cap prefixes)/' "$SCORES/TRUNCATION.md"

say "collect"
python -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.collect_campaign_scores --root "$ROOT" --out "$SCORES"
cp "$SCORES/campaign_battery_scores.json" "$RLVR/eval_scores/thinking_t07_cap12k/campaign_battery_scores.json"
cp "$SCORES/campaign_battery_scores.csv" "$RLVR/eval_scores/thinking_t07_cap12k/campaign_battery_scores.csv"

say "analyse (HEADLINE)"
python -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.analyse_campaign_battery --root "$ROOT" --mode thinking --out "$SCORES/HEADLINE.md" > /dev/null

say "paired 4k vs 12k comparison"
python experiments/prior_coins/rlvr_thinking_malformed_v1/compare_caps.py "$ROOT" > /dev/null
cp experiments/prior_coins/rlvr_thinking_malformed_v1/results/cap12k_comparison.md "$SCORES/CAP_COMPARISON.md"
cp experiments/prior_coins/rlvr_thinking_malformed_v1/results/cap12k_comparison.json "$SCORES/CAP_COMPARISON.json"

say "one-/two-run and clause slices"
python $RLVR/collect_run_count_scores.py --sweep thinking-t07-cap12k --revision main --by-clause \
  --out "$RLVR/eval_scores" --reference "$SCORES/campaign_battery_scores.json"

say "figures: pooled gallery"
python $RLVR/plot_eval_trajectories.py --scores "$SCORES/campaign_battery_scores.json" --out "$FIGS" --decoding-note "$NOTE" > "$FIGS/.plot.log"
for rc in onerun tworun; do
  say "figures: $rc"
  python $RLVR/plot_eval_trajectories.py --scores "$RLVR/eval_scores/thinking_t07_cap12k_campaign_battery_scores_${rc}.json" \
    --out "$FIGS/$rc" --decoding-note "$NOTE" >> "$FIGS/.plot.log"
  for scores in "$RLVR"/eval_scores/run_count_clauses/thinking-t07-cap12k/${rc}/*.json; do
    clause=$(basename "$scores" .json)
    python $RLVR/plot_eval_trajectories.py --scores "$scores" --out "$FIGS/$rc/$clause" --decoding-note "$NOTE" >> "$FIGS/.plot.log"
  done
done
rm -f "$FIGS/.plot.log"
find "$FIGS" -name "*.png" | wc -l | xargs echo "PNG figures:"

if [ "$UPLOAD" = "--upload" ]; then
  say "upload eval_scores to $PREFIX/eval_scores"
  python - "$HUB_REPO" "$SCORES" "$PREFIX/eval_scores" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, folder, dest = sys.argv[1:4]
api = HfApi(token=os.environ["HF_WRITE_TOKEN_ARCADIA"])
api.upload_folder(repo_id=repo, repo_type="model", folder_path=folder, path_in_repo=dest,
                  commit_message="thinking-t07-cap12k eval_scores (re-score of the six continued endpoints)")
print("UPLOAD_OK", dest, [f.rsplit('/',1)[-1] for f in api.list_repo_files(repo) if f.startswith(dest + "/")])
PY
fi
say "done: $SCORES  $FIGS"
