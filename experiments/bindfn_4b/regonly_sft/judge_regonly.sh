#!/usr/bin/env bash
# Crab-side: judge-score the `describe` rows of the regonly hard-eval gens.
#
#   bash judge_regonly.sh /workspace/bindfn4b_backup/regonly_sft/hardeval_gens/*.jsonl
#
# grading.py's describe branch is only a weak string-match lower bound; the
# real scorer is eval/judge_describe.py (deepseek-v4-flash via OpenRouter,
# translate description -> lambda -> run on 20 holdout probe_xs). Judge output
# lands under results/describe_judge/<spec>/ and summarize.py overrides the
# weak column with it. Drop rate above 10% raises — rerun (the .judge_cache
# makes a rerun nearly free) rather than accepting a deflated score.
set -euo pipefail
cd /workspace/science-of-midtraining
set -a; . ./.env; set +a
OUT=experiments/bindfn_4b/regonly_sft/results/describe_judge
mkdir -p "$OUT"
for gens in "$@"; do
  spec=$(basename "$gens" .jsonl)
  echo "== judging $spec"
  uv run --no-project --with httpx python \
      experiments/bindfn_4b/eval/judge_describe.py \
      --gens "$gens" \
      --items experiments/bindfn_4b/eval/data/hard_eval.jsonl \
      --out-dir "$OUT/${spec#_workspace_ck_}"
done
echo JUDGE_DONE
