#!/usr/bin/env bash
# Charter-recall probes at four points of the trajectory, one per GPU.
#
#   midtrain_381   end of midtraining, BEFORE instruct-tuning   (base model)
#   pre_aft        end of Dolci SFT, no adapter
#   aft_256        agreement-only AFT, 1 epoch
#   aft_512        agreement-only AFT, 2 epochs
#
# Four endpoints, four cards, so this is one engine per GPU with no staging
# needed -- unlike eval_sharded.sh, nothing here shares a prompt download worth
# warming first (the prompt set is 84 local rows).
#
# The `agreement` cell supplies the AFT points deliberately: it carries no
# charter/coin conflict labels, so the trajectory isolates what adversarial
# fine-tuning does to recall without the label-flip manipulation on top.
#
# Safe to re-run: each endpoint skips itself if RECALL_COMPLETE.json exists.
set -uo pipefail

ARM=${1:?usage: recall_sharded.sh <arm> [root] [endpoints]}
ROOT=${2:-/workspace/final_v1}
#: comma-separated, passed by chain.py so the midtrain step tracks the
#: profile's schedule; the default is the as-run gemma3_12b_50m set.
ENDPOINTS=${3:-midtrain_381,pre_aft,aft_256,aft_512}
REPO=${REPO:-/workspace/scimt}
P="$ROOT/$ARM"

export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)

EVAL_PYTHON=${FINAL_V1_EVAL_PYTHON:-/workspace/venv-dispatch-eval/bin/python}
RECALL=$REPO/experiments/prior_coins/dispatch_final_v1/pod/recall_eval.py
PROMPTS=${RECALL_PROMPTS:-$P/data/recall/prompts}

[ -d "$PROMPTS" ] || { echo "no recall prompts at $PROMPTS"; exit 1; }
mkdir -p "$P/recall"

echo "[$(date -u +%T)] $ARM: recall endpoints [$ENDPOINTS] across the GPUs"
gpu=0
pids=()
for ep in ${ENDPOINTS//,/ }; do
  "$EVAL_PYTHON" "$RECALL" --arm "$ARM" --endpoint "$ep" --gpu "$gpu" \
    --root "$ROOT" \
    --prompts "$PROMPTS" --out "$P/recall/$ep" --work "$P/recall-work-gpu$gpu" \
    >> "$P/recall/shard-$ep.log" 2>&1 &
  pids+=($!)
  echo "  $ep -> GPU $gpu (pid ${pids[-1]})"
  gpu=$((gpu + 1))
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

n=$(find "$P/recall" -name RECALL_COMPLETE.json | wc -l)
want=$(echo "$ENDPOINTS" | tr ',' '\n' | wc -l)
echo "[$(date -u +%T)] $ARM: recall shards done (fail=$fail), $n/$want endpoints"
exit "$fail"
