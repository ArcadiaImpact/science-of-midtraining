#!/usr/bin/env bash
# One arm, one GPU, THINKING mode: the step-0 anchor (LoRA off) then that arm
# pinned adapters (LoRA on). Waits only on ITS OWN graft marker, so the first
# arm starts while the last graft is still downloading.
#
# Traps this is written around:
#  - pgrep -f self-matches the ssh command line, so readiness is a marker FILE;
#  - a crashed eval hangs in vLLM teardown holding ~118 GiB, so every sweep is
#    wrapped in timeout;
#  - three vLLM engines on one host collide unless each gets its own VLLM_PORT.
set -uo pipefail
. /workspace/hf.env
ARM=$1
GPU=$2
PY=/workspace/venvs/dispatch-rlvr-rl/bin/python
OUT=/workspace/evals-campaign-battery/thinking/$ARM
PARENT=/workspace/graft-dl/grafts/$ARM
DATA=/workspace/eval_data
mkdir -p "$OUT"
cd /workspace/scimt-dispatch-rlvr-gemma4-26b-v1 || exit 11

while [ ! -f "/workspace/grafts/$ARM.done" ]; do sleep 30; done
test -f "$PARENT/config.json" || { echo "FATAL: no graft at $PARENT"; exit 12; }

export VLLM_PORT=$((8000 + GPU * 100))
export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false

for PHASE in anchor adapters; do
  MARK="$OUT/.$PHASE.done"
  if [ -f "$MARK" ]; then echo "SKIP $ARM/$PHASE"; continue; fi
  echo "=== $(date -u +%H:%M:%S) $ARM/$PHASE gpu=$GPU ==="
  # tier defaults to trained; tier=all with mode=thinking raises by design.
  # Sampling params are the RLVR thinking defaults, deliberately unchanged:
  # 4096-token cap, temperature 0. Truncation is measured, not fixed.
  timeout 11h "$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.campaign_sweep \
    mode=thinking tier=trained workers=48 \
    parent_model="$PARENT" \
    endpoints="/workspace/plans/thinking/$ARM.$PHASE.json" \
    output_dir="$OUT" data_dir="$DATA"
  rc=$?
  if [ "$rc" -ne 0 ]; then echo "FAIL $ARM/$PHASE rc=$rc"; exit "$rc"; fi
  touch "$MARK"
done
touch "$OUT/.arm.done"
echo "ARM_DONE $ARM"
