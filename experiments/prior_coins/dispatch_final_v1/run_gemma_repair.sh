#!/usr/bin/env bash
set -euo pipefail
cd /workspace/scimt-1c
export PYTHONPATH=/workspace/scimt-1c:/workspace/scimt-1c/src
export HF_HOME=/workspace/hf-final-v1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
WORKER=${1:?worker ID}
MODEL=${WORKER#*-}; MODEL=${MODEL%-*}
export FINAL_V1_PROFILE=gemma3_${MODEL}_5m
exec python3 -m experiments.prior_coins.dispatch_final_v1.gemma_repair_wait \
  --worker "$WORKER" --plan /workspace/grid-repair-input/repair-plan-52cells.json \
  --data /workspace/grid-input/data-validated \
  --predecessor /workspace/gemma-grid/"$WORKER" \
  --root /workspace/gemma-grid-repair/"$WORKER" \
  --publish-repo arcadia-impact/scimt-dispatch-gemma-${MODEL}-aft-grid-v2
