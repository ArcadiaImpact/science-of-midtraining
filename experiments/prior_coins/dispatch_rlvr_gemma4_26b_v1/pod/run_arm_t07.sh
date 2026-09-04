#!/usr/bin/env bash
# One arm, one GPU, THINKING mode at temperature 0.7 (SAMPLED, not greedy).
#
# Run order is 768, 0, 512, 256 per Sid: the widest contrast first, so a setup
# problem shows up on the two most informative endpoints rather than after six
# hours. That costs one extra engine boot per arm (~5 min), because an engine is
# either LoRA-on or LoRA-off and cannot be both -- so the adapter phase has to
# be split around the step-0 anchor:
#
#   p1      adapters, step 768        (LoRA ON)
#   anchor  step 0                    (LoRA OFF)
#   p3      adapters, steps 512, 256  (LoRA ON)
#
# Traps this is written around: pgrep -f self-matches the ssh command line, so
# readiness is a marker FILE; a crashed eval hangs in vLLM teardown holding
# ~118 GiB, so every sweep is wrapped in timeout; three vLLM engines on one host
# collide unless each gets its own VLLM_PORT.
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

while [ ! -f "/workspace/grafts/$ARM.done" ]; do sleep 30; done
test -f "$PARENT/config.json" || { echo "FATAL: no graft at $PARENT"; exit 12; }

export VLLM_PORT=$((8000 + GPU * 100))
export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false

for PHASE in p1 anchor p3; do
  MARK="$OUT/.$PHASE.done"
  if [ -f "$MARK" ]; then echo "SKIP $ARM/$PHASE"; continue; fi
  PLAN="/workspace/plans/t07/$ARM.$PHASE.json"
  if [ ! -s "$PLAN" ]; then echo "FATAL: no plan at $PLAN"; exit 13; fi
  echo "=== $(date -u +%H:%M:%S) $ARM/$PHASE gpu=$GPU ==="
  timeout 6h "$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.campaign_sweep \
    mode=thinking tier=trained workers=48 \
    temperature=0.7 seed=20260904 \
    parent_model="$PARENT" \
    endpoints="$PLAN" \
    output_dir="$OUT" data_dir="$DATA"
  rc=$?
  if [ "$rc" -ne 0 ]; then echo "FAIL $ARM/$PHASE rc=$rc"; exit "$rc"; fi
  touch "$MARK"
done
touch "$OUT/.arm.done"
echo "ARM_DONE $ARM"
