#!/usr/bin/env bash
# One provisioned single-H200 charter-dominant queue (gemma3_27b_190m, one arm).
# Usage: run_gemma_charter_dominant.sh <worker: charter-27b-cd|control-27b-cd|coin-27b-cd>
# Expects /workspace/cd-input/{plan.json,READY.json,data/,data-receipts/shared-data.json}
# and the repo snapshot at /workspace/scimt. HF_TOKEN must already be exported.
set -euo pipefail
cd /workspace/scimt
export PYTHONPATH=/workspace/scimt:/workspace/scimt/src
export HF_HOME=/workspace/hf-final-v1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export FINAL_V1_PROFILE=gemma3_27b_190m
WORKER=${1:?worker ID}
ROOT=/workspace/gemma-cd/${WORKER}
mkdir -p "$ROOT/data-receipts"
cp /workspace/cd-input/data-receipts/shared-data.json "$ROOT/data-receipts/"
if ! test -f /workspace/gemma-setup-complete; then
  bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh > /workspace/gemma-setup.log 2>&1
  touch /workspace/gemma-setup-complete
fi
exec python3 -m experiments.prior_coins.dispatch_final_v1.gemma_charter_dominant worker \
  --plan /workspace/cd-input/plan.json --data /workspace/cd-input/data \
  --root "$ROOT" --worker "$WORKER" --execute --approved-launch
