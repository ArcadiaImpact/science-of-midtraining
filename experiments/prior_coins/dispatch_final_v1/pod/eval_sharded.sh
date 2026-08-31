#!/usr/bin/env bash
# Sample one arm's 9 endpoints across the profile's available GPUs.
#
# The first version of phase_eval ran endpoints serially, on the reasoning that
# vLLM wants a whole device. That is true PER ENGINE -- each needs its own ~24 GB
# model copy plus KV cache, so four will not fit on one card -- but the pod has
# multiple cards, so one engine per GPU is fine. When cells outnumber cards,
# each card drains its assigned cells sequentially.
#
# Sharding is by CELL, which is the natural unit: pod_generate_multi holds one
# resident base and sweeps that cell's two adapters through it, so a cell is
# already "one engine, many endpoints". pre_aft runs first and alone, because it
# also warms the prompt download that the cells then reuse.
#
# Safe to re-run: every writer skips a prompt set whose output file is already
# complete, so a relaunch resumes rather than repeats.
set -uo pipefail

ARM=${1:?usage: eval_sharded.sh <arm>}
ROOT=${2:-/workspace/final_v1}
REPO=${REPO:-/workspace/scimt}
P="$ROOT/$ARM"
cd "$REPO"

export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)

EVAL_PYTHON=${FINAL_V1_EVAL_PYTHON:-python3}
CONTRACTS_DIR=$REPO/experiments/prior_coins/dispatch_final_v1
read -r N_GPUS TP DOLCI_STEPS FAMILY < <(
  PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
  'import contracts; print(contracts.N_GPUS, contracts.EVAL_TENSOR_PARALLEL_SIZE, contracts.DOLCI_STEPS, contracts.MODEL_FAMILY)')

[ "$N_GPUS" -ge 1 ] || { echo "profile n_gpus must be positive"; exit 1; }
[ "$TP" -ge 1 ] && [ $((N_GPUS % TP)) -eq 0 ] || {
  echo "eval tensor parallel size $TP does not divide $N_GPUS GPUs"; exit 1;
}
N_GROUPS=$((N_GPUS / TP))

if [ "$FAMILY" = glm45_air ]; then
  PARENT="$P/dolci/consolidated/checkpoint-$DOLCI_STEPS"
else
  PARENT="$P/dolci/checkpoints/checkpoint-48"
fi
[ -d "$PARENT" ] || { echo "no parent at $PARENT"; exit 1; }

gpu_group() {
  local slot=$1 group="" index
  local start=$((slot * TP))
  for ((index=start; index<start+TP; index++)); do
    group=${group:+$group,}$index
  done
  echo "$group"
}

run_one() {  # cell gpu
  CUDA_VISIBLE_DEVICES=$2 "$EVAL_PYTHON" experiments/prior_coins/dispatch_final_v1/pod/evaluate.py \
    --arm "$ARM" --parent "$PARENT" \
    --aft-root "$P/aft" --aft-data "$P/data/aft" \
    --out "$P/eval" --work "$P/xgen-gpu$2" --only "$1" \
    >> "$P/eval/shard-$1.log" 2>&1
}

mkdir -p "$P/eval"

# pre_aft alone on GPU 0: it fetches the 18 prompt sets the cells then reuse, so
# running it first avoids four processes racing the same downloads.
FIRST_GROUP=$(gpu_group 0)
echo "[$(date -u +%T)] $ARM: pre_aft on GPU $FIRST_GROUP"
run_one pre_aft "$FIRST_GROUP" || echo "pre_aft FAILED (continuing; cells are independent)"

# pre_aft ran on GPU 0 and a cell is about to reuse it. A process that has
# EXITED can still hold CUDA memory for a few seconds, and vLLM sizes its cache
# off free memory at startup, so launching immediately is how GPU 0 OOMs while
# GPUs 1-3 are fine. Wait for the card to actually drain.
echo "[$(date -u +%T)] $ARM: waiting for GPU $FIRST_GROUP to drain"
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$FIRST_GROUP" \
    | awk '{sum += $1} END {print sum + 0}')
  [ "${used:-99999}" -lt $((2000 * TP)) ] && break
  sleep 5
done
echo "[$(date -u +%T)] $ARM: GPU $FIRST_GROUP at ${used}MiB total"

CELLS=(agreement mixed_charter mixed_coin charter_only)
N_CELLS=${#CELLS[@]}
N_WORKERS=$((N_GROUPS < N_CELLS ? N_GROUPS : N_CELLS))
echo "[$(date -u +%T)] $ARM: $N_CELLS cells across $N_WORKERS/$N_GROUPS TP groups ($N_GPUS GPUs)"
pids=()
offset=0
for ((gpu=0; gpu<N_WORKERS; gpu++)); do
  size=$((N_CELLS / N_WORKERS))
  if ((gpu < N_CELLS % N_WORKERS)); then
    size=$((size + 1))
  fi
  shard=("${CELLS[@]:offset:size}")
  group=$(gpu_group "$gpu")
  (
    # One worker per GPU: cells assigned to the same card run sequentially so
    # two resident vLLM engines never compete for that card's memory.
    for cell in "${shard[@]}"; do
      run_one "$cell" "$group" || exit 1
    done
  ) &
  pids+=($!)
  echo "  gpu $group <- ${shard[*]} (pid ${pids[-1]})"
  offset=$((offset + size))
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

n=$(find "$P/eval" -name '*__*.jsonl' | wc -l)
echo "[$(date -u +%T)] $ARM: shards done (fail=$fail), $n/162 response files"
exit "$fail"
