#!/usr/bin/env bash
# Charter-cost sweep for one arm. The profile owns GPU count; with the as-run
# four-GPU profile the balanced contiguous split is exactly D4's 3/2/2/2.
set -uo pipefail

ARM=${1:?usage: costsweep_sharded.sh <arm> [profile-root]}
ROOT=${2:-/workspace/final_v1}
REPO=${REPO:-/workspace/scimt}
P="$ROOT/$ARM"

export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)

EVAL_PYTHON=${FINAL_V1_EVAL_PYTHON:-/workspace/venv-dispatch-eval/bin/python}
CONTRACTS_DIR=$REPO/experiments/prior_coins/dispatch_final_v1
RUNNER=$CONTRACTS_DIR/pod/costsweep_eval.py
PROMPTS=${COSTSWEEP_PROMPTS:-$P/costsweep/data/prompts/costsweep.jsonl}
N_GPUS=$(PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
  'import contracts; print(contracts.N_GPUS)')

[ -f "$PROMPTS" ] || { echo "no costsweep prompts at $PROMPTS"; exit 1; }
[ "$N_GPUS" -ge 1 ] || { echo "profile n_gpus must be positive"; exit 1; }

ENDPOINTS=(
  pre_aft
  agreement-step256 agreement-step512
  mixed_charter-step256 mixed_charter-step512
  mixed_coin-step256 mixed_coin-step512
  charter_only-step256 charter_only-step512
)
N_ENDPOINTS=${#ENDPOINTS[@]}
[ "$N_GPUS" -le "$N_ENDPOINTS" ] || {
  echo "profile has $N_GPUS GPUs but only $N_ENDPOINTS costsweep endpoints"
  exit 1
}

mkdir -p "$P/costsweep"
echo "[$(date -u +%T)] $ARM: costsweep across $N_GPUS profile GPUs"
pids=()
offset=0
for ((gpu=0; gpu<N_GPUS; gpu++)); do
  size=$((N_ENDPOINTS / N_GPUS))
  if ((gpu < N_ENDPOINTS % N_GPUS)); then
    size=$((size + 1))
  fi
  shard=""
  for ((index=offset; index<offset+size; index++)); do
    shard=${shard:+$shard,}${ENDPOINTS[$index]}
  done
  offset=$((offset + size))
  "$EVAL_PYTHON" "$RUNNER" --arm "$ARM" --gpu "$gpu" --prompts "$PROMPTS" \
    --root "$ROOT" --out "$P/costsweep" --work "$P/costsweep-work-gpu$gpu" \
    --endpoints "$shard" >> "$P/costsweep/shard-gpu$gpu.log" 2>&1 &
  pids+=($!)
  echo "  gpu $gpu <- $shard (pid ${pids[-1]})"
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

n=$(find "$P/costsweep" -name COSTSWEEP_COMPLETE.json | wc -l)
echo "[$(date -u +%T)] $ARM: costsweep shards done (fail=$fail), $n/$N_ENDPOINTS endpoints"
exit "$fail"
