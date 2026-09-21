#!/usr/bin/env bash
# One provisioned, account-owned single-GPU production queue.
set -euo pipefail
cd /workspace/scimt
export PYTHONPATH=/workspace/scimt:/workspace/scimt/src
export HF_HOME=/workspace/hf-final-v1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
WORKER=${1:?worker ID}
MODEL=${WORKER#*-}; MODEL=${MODEL%-*}
export FINAL_V1_PROFILE=gemma3_${MODEL}_5m
OUT_REPO=arcadia-impact/scimt-dispatch-gemma-${MODEL}-aft-grid-v2
ROOT=/workspace/gemma-grid/${WORKER}
mkdir -p "$ROOT/data-receipts"
cp /workspace/grid-input/publish-${MODEL}/data-receipts/shared-data.json "$ROOT/data-receipts/"
if ! test -f /workspace/gemma-setup-complete; then
  bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh > /workspace/gemma-setup.log 2>&1
  touch /workspace/gemma-setup-complete
fi
exec python3 -m experiments.prior_coins.dispatch_final_v1.gemma_grid_run worker \
  --plan /workspace/grid-input/grid-plan-12workers.json \
  --data /workspace/grid-input/data-validated --root "$ROOT" --worker "$WORKER" \
  --publish-repo "$OUT_REPO" --execute
