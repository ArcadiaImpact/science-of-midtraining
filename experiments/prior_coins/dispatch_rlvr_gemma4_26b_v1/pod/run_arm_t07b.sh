#!/usr/bin/env bash
# Same as run_arm_t07.sh but with the two throughput knobs applied:
#   max_model_len 7168 -> 5120   (requirement is 953 + 4096 = 5049)
#   gpu_memory_utilization 0.82 -> 0.92
#
# Observed concurrency at the old geometry was 36.41x, which vLLM computes as
# (KV cache tokens) / max_model_len = 260,971 / 7,168. The new geometry should
# give roughly 322k / 5,120 ~= 63x, i.e. ~1.7x more sequences in flight.
#
# Neither knob changes the model mathematically -- max_model_len bounds
# scheduler admission and gpu_memory_utilization sizes the KV cache -- and
# campaign_sweep validates the window against the LONGEST rendered prompt
# before booting, so it cannot silently shorten completions.
#
# Phase markers make this a resume, not a restart: p1 (step 768) already ran at
# the OLD geometry and is skipped. Each endpoint records its own geometry.
set -uo pipefail
. /workspace/hf.env
ARM=$1
GPU=$2
PY=/workspace/venvs/dispatch-rlvr-rl/bin/python
OUT=/workspace/evals-campaign-battery/thinking-t07/$ARM
PARENT=/workspace/graft-dl/grafts/$ARM
DATA=/workspace/eval_data
mkdir -p "$OUT"
cd /workspace/scimt-dispatch-rlvr-gemma4-26b-v1 || exit 11

test -f "$PARENT/config.json" || { echo "FATAL: no graft at $PARENT"; exit 12; }

export VLLM_PORT=$((8000 + GPU * 100))
export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false

for PHASE in p1 anchor p3; do
  MARK="$OUT/.$PHASE.done"
  if [ -f "$MARK" ]; then echo "SKIP $ARM/$PHASE (already done)"; continue; fi
  PLAN="/workspace/plans/t07/$ARM.$PHASE.json"
  if [ ! -s "$PLAN" ]; then echo "FATAL: no plan at $PLAN"; exit 13; fi
  echo "=== $(date -u +%H:%M:%S) $ARM/$PHASE gpu=$GPU FAST_GEOMETRY ==="
  timeout 6h "$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.campaign_sweep \
    mode=thinking tier=trained workers=48 \
    temperature=0.7 seed=20260904 \
    max_model_len=5120 gpu_memory_utilization=0.92 \
    parent_model="$PARENT" \
    endpoints="$PLAN" \
    output_dir="$OUT" data_dir="$DATA"
  rc=$?
  if [ "$rc" -ne 0 ]; then echo "FAIL $ARM/$PHASE rc=$rc"; exit "$rc"; fi
  touch "$MARK"
done
touch "$OUT/.arm.done"
echo "ARM_DONE $ARM"
