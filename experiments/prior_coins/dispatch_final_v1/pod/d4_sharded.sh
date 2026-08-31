#!/usr/bin/env bash
# D4 'withheld records' for one arm, balanced over the profile's GPUs.
#
# The standalone runner puts all 9 endpoints on ONE engine, which is right when
# three arms share a 3-GPU pod (one arm per card, ~22 min each in parallel).
# In the chain a pod owns ONE arm, so the profile-sized split keeps its cards
# busy. Each GPU receives one comma-separated shard and one resident engine.
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
CONTRACTS_DIR=$REPO/experiments/prior_coins/dispatch_final_v1
RUNNER=$CONTRACTS_DIR/pod/d4_eval.py
ITEMS=${D4_ITEMS:-$P/data/d4/d4_inforequest.jsonl}
read -r N_GPUS TP _DOLCI_STEPS _FAMILY < <(
  PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
  'import contracts; print(contracts.N_GPUS, contracts.EVAL_TENSOR_PARALLEL_SIZE, contracts.DOLCI_STEPS, contracts.MODEL_FAMILY)')

[ -f "$ITEMS" ] || { echo "no D4 items at $ITEMS"; exit 1; }
[ "$N_GPUS" -ge 1 ] || { echo "profile n_gpus must be positive"; exit 1; }
[ "$TP" -ge 1 ] && [ $((N_GPUS % TP)) -eq 0 ] || exit 1
N_GROUPS=$((N_GPUS / TP))
mkdir -p "$P/d4"

gpu_group() {
  local slot=$1 group="" index
  local start=$((slot * TP))
  for ((index=start; index<start+TP; index++)); do
    group=${group:+$group,}$index
  done
  echo "$group"
}

ENDPOINTS=(
  pre_aft
  agreement-step256 agreement-step512
  mixed_charter-step256 mixed_charter-step512
  mixed_coin-step256 mixed_coin-step512
  charter_only-step256 charter_only-step512
)
N_ENDPOINTS=${#ENDPOINTS[@]}
N_WORKERS=$((N_GROUPS < N_ENDPOINTS ? N_GROUPS : N_ENDPOINTS))

echo "[$(date -u +%T)] $ARM: D4 across $N_WORKERS/$N_GROUPS TP groups ($N_GPUS GPUs)"
pids=()
offset=0
for ((gpu=0; gpu<N_WORKERS; gpu++)); do
  size=$((N_ENDPOINTS / N_WORKERS))
  if ((gpu < N_ENDPOINTS % N_WORKERS)); then
    size=$((size + 1))
  fi
  shard=""
  for ((index=offset; index<offset+size; index++)); do
    shard=${shard:+$shard,}${ENDPOINTS[$index]}
  done
  offset=$((offset + size))
  group=$(gpu_group "$gpu")
  "$EVAL_PYTHON" "$RUNNER" --arm "$ARM" --gpu "$group" --items "$ITEMS" \
    --root "$ROOT" \
    --out "$P/d4" --work "$P/d4-work-gpu$gpu" --endpoints "$shard" \
    >> "$P/d4/shard-gpu$gpu.log" 2>&1 &
  pids+=($!)
  echo "  gpu $group <- $shard (pid ${pids[-1]})"
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

n=$(find "$P/d4" -name D4_COMPLETE.json | wc -l)
echo "[$(date -u +%T)] $ARM: D4 shards done (fail=$fail), $n/9 endpoints"
exit "$fail"
