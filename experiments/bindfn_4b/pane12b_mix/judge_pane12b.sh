#!/usr/bin/env bash
# Crab-side: judge-score the `describe` rows of the pane12b_mix eval gens.
#
#   bash judge_pane12b.sh /workspace/bindfn4b_backup/pane12b_mix/gens/*.jsonl
#
# grading.py's describe branch is only a weak string-match lower bound; the
# real scorer is eval/judge_describe.py (deepseek-v4-flash via OpenRouter:
# translate the description -> a lambda -> run it in the subprocess sandbox on
# the item's 20 holdout probe_xs). Judge output lands under
# results/describe_judge/<spec>/ and summarize.py / paired_stats.py override
# the weak column with it. A drop rate above 10% raises — rerun (the
# .judge_cache makes a rerun nearly free) rather than accept a deflated score.
#
# Only the `full` suite carries describe rows; light-suite gens are skipped
# automatically (no describe rows -> the judge writes an empty score file).
set -euo pipefail
cd /workspace/science-of-midtraining
set -a; . ./.env; set +a
OUT=experiments/bindfn_4b/pane12b_mix/results/describe_judge
ITEMS=experiments/bindfn_4b/pane12b_mix/data/hard_eval.jsonl
mkdir -p "$OUT"
for gens in "$@"; do
  spec=$(basename "$gens" .jsonl)
  grep -q '"eval_type": "describe"' "$gens" || { echo "-- $spec: no describe rows"; continue; }
  echo "== judging $spec"
  uv run --no-project --with httpx python \
      experiments/bindfn_4b/eval/judge_describe.py \
      --gens "$gens" --items "$ITEMS" --out-dir "$OUT/$spec"
done
echo JUDGE_DONE
