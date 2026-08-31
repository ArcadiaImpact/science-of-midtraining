#!/usr/bin/env bash
# Sample one arm's 9 endpoints across all 4 GPUs instead of one at a time.
#
# The first version of phase_eval ran endpoints serially, on the reasoning that
# vLLM wants a whole device. That is true PER ENGINE -- each needs its own ~24 GB
# model copy plus KV cache, so four will not fit on one card -- but the pod has
# four cards, so the conclusion was wrong: one engine per GPU is fine, and is
# what the AFT phase already does. Serial cost ~4x the wall clock.
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

PARENT="$P/dolci/checkpoints/checkpoint-48"
[ -d "$PARENT" ] || { echo "no parent at $PARENT"; exit 1; }

run_one() {  # cell gpu
  CUDA_VISIBLE_DEVICES=$2 python3 experiments/prior_coins/dispatch_final_v1/pod/evaluate.py \
    --arm "$ARM" --parent "$PARENT" \
    --aft-root "$P/aft" --aft-data "$P/data/aft" \
    --out "$P/eval" --work "$P/xgen-gpu$2" --only "$1" \
    >> "$P/eval/shard-$1.log" 2>&1
}

mkdir -p "$P/eval"

# pre_aft alone on GPU 0: it fetches the 18 prompt sets the cells then reuse, so
# running it first avoids four processes racing the same downloads.
echo "[$(date -u +%T)] $ARM: pre_aft on GPU 0"
run_one pre_aft 0 || echo "pre_aft FAILED (continuing; cells are independent)"

# pre_aft ran on GPU 0 and a cell is about to reuse it. A process that has
# EXITED can still hold CUDA memory for a few seconds, and vLLM sizes its cache
# off free memory at startup, so launching immediately is how GPU 0 OOMs while
# GPUs 1-3 are fine. Wait for the card to actually drain.
echo "[$(date -u +%T)] $ARM: waiting for GPU 0 to drain"
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 5
done
echo "[$(date -u +%T)] $ARM: GPU 0 at ${used}MiB"

echo "[$(date -u +%T)] $ARM: 4 cells across GPUs 0-3"
gpu=0
pids=()
for cell in agreement mixed_charter mixed_coin charter_only; do
  run_one "$cell" "$gpu" &
  pids+=($!)
  echo "  $cell -> GPU $gpu (pid ${pids[-1]})"
  gpu=$((gpu + 1))
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

n=$(find "$P/eval" -name '*__*.jsonl' | wc -l)
echo "[$(date -u +%T)] $ARM: shards done (fail=$fail), $n/162 response files"
exit "$fail"
