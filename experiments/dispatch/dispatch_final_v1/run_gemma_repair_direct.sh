#!/usr/bin/env bash
# Physical relocation of an unchanged four-cell logical #1c queue.
set -euo pipefail
cd /workspace/scimt
export PYTHONPATH=/workspace/scimt:/workspace/scimt/src
export HF_HOME=/workspace/hf-final-v1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export FINAL_V1_PROFILE=gemma3_27b_5m
WORKER=${1:?logical worker ID}
test -f /workspace/gemma-grid-repair/"$WORKER"/TRANSFER_IN.json
if ! test -f /workspace/gemma-setup-complete; then
  bash experiments/dispatch/dispatch_final_v1/pod/setup.sh > /workspace/gemma-setup.log 2>&1
  touch /workspace/gemma-setup-complete
fi
exec python3 -m experiments.dispatch.dispatch_final_v1.gemma_grid_run worker \
  --plan /workspace/grid-repair-input/repair-plan-52cells.json \
  --data /workspace/grid-input/data-validated \
  --root /workspace/gemma-grid-repair/"$WORKER" --worker "$WORKER" \
  --publish-repo arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2 --execute
