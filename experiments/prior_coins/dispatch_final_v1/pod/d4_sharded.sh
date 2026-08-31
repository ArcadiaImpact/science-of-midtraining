#!/usr/bin/env bash
# D4 'withheld records' for one arm, its 9 endpoints spread over the 4 GPUs.
#
# The standalone runner puts all 9 endpoints on ONE engine, which is right when
# three arms share a 3-GPU pod (one arm per card, ~22 min each in parallel).
# In the chain a pod owns ONE arm and has four cards, so that layout would idle
# three of them. Sharding costs 4 model loads instead of 1 (~4 min each) and
# buys ~2x: ~10 min instead of ~22.
#
# Split is by count, not by cell, because every endpoint costs the same here
# (256 items, prefill-bound) -- unlike eval_sharded.sh, where sharding by cell
# is what lets one resident base sweep that cell's adapters.
#
# Safe to re-run: each endpoint skips itself if D4_COMPLETE.json exists.
set -uo pipefail

ARM=${1:?usage: d4_sharded.sh <arm>}
ROOT=${2:-/workspace/final_v1}
REPO=${REPO:-/workspace/scimt}
P="$ROOT/$ARM"

export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)

EVAL_PYTHON=${FINAL_V1_EVAL_PYTHON:-/workspace/venv-dispatch-eval/bin/python}
RUNNER=$REPO/experiments/prior_coins/dispatch_final_v1/pod/d4_eval.py
ITEMS=${D4_ITEMS:-$P/data/d4/d4_inforequest.jsonl}

[ -f "$ITEMS" ] || { echo "no D4 items at $ITEMS"; exit 1; }
mkdir -p "$P/d4"

# 9 endpoints over 4 GPUs: 3,2,2,2
SHARD0="pre_aft,agreement-step256,agreement-step512"
SHARD1="mixed_charter-step256,mixed_charter-step512"
SHARD2="mixed_coin-step256,mixed_coin-step512"
SHARD3="charter_only-step256,charter_only-step512"

echo "[$(date -u +%T)] $ARM: D4 across GPUs 0-3"
pids=()
gpu=0
for shard in "$SHARD0" "$SHARD1" "$SHARD2" "$SHARD3"; do
  "$EVAL_PYTHON" "$RUNNER" --arm "$ARM" --gpu "$gpu" --items "$ITEMS" \
    --root "$ROOT" \
    --out "$P/d4" --work "$P/d4-work-gpu$gpu" --endpoints "$shard" \
    >> "$P/d4/shard-gpu$gpu.log" 2>&1 &
  pids+=($!)
  echo "  gpu $gpu <- $shard (pid ${pids[-1]})"
  gpu=$((gpu + 1))
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

n=$(find "$P/d4" -name D4_COMPLETE.json | wc -l)
echo "[$(date -u +%T)] $ARM: D4 shards done (fail=$fail), $n/9 endpoints"
exit "$fail"
