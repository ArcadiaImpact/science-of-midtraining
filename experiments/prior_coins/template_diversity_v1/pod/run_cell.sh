#!/usr/bin/env bash
# Run ONE template-diversity cell (one substrate) on this pod, end to end.
#
#   run_cell.sh <label> <parent_prefix> <parent_revision> [parent_repo]
#
# e.g. run_cell.sh charter_real_4x sft_4epoch/charter/checkpoint-48 \
#        527f0b6cc0ea117e7c9e89e82221163654bd50db
#
# Mirrors run_wave_worklist.sh but: single cell, the template-diversity data
# prefix, and checkpoint upload ON (persisting the LoRAs is a run goal).
set -uo pipefail

LABEL="${1:?usage: run_cell.sh <label> <parent_prefix> <parent_revision> [parent_repo]}"
PARENT_PREFIX="${2:?}"
REVISION="${3:?}"
PARENT_REPO="${4:-jbostock/scimt-dispatch-midtrained-sft-v1}"
REPO=/workspace/scimt-prior-coins
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-wave
export HF_HUB_ENABLE_HF_TRANSFER=1
export WAVE_ROOT=/workspace/wave
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WAVE_ROOT/status"
STATUS="$WAVE_ROOT/status/$LABEL"
if [ -f "$STATUS.done" ]; then echo "[skip] $LABEL already done"; exit 0; fi
echo "=== CELL $LABEL ($(date -u +%H:%M:%S)) parent=$PARENT_PREFIX"

if [ ! -f "$WAVE_ROOT/PREPARE_DONE.json" ]; then
  echo "--- prepare parent $PARENT_PREFIX"
  rm -rf "$WAVE_ROOT/parent" "$WAVE_ROOT/_parent_staging"
  if ! python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_prepare.py" \
      --label "$LABEL" --parent-repo "$PARENT_REPO" \
      --parent-prefix "$PARENT_PREFIX" \
      --parent-revision "$REVISION" \
      --data-prefix extensions/template_diversity_v1/data; then
    echo "PREPARE_FAILED $LABEL"; echo "prepare" > "$STATUS.failed"; exit 1
  fi
fi

if python3 "$REPO/experiments/prior_coins/template_diversity_v1/pod/chain.py" \
    --label "$LABEL" --parent-label "$LABEL" \
    --parent-repo "$PARENT_REPO" \
    --parent-prefix "$PARENT_PREFIX" --parent-revision "$REVISION" \
    --dataset agreement --remote-root extensions/template_diversity_v1 \
    --version template_diversity_v1; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$STATUS.done"
  echo "=== CELL DONE $LABEL ($(date -u +%H:%M:%S))"
else
  echo "chain" > "$STATUS.failed"
  echo "=== CELL FAILED $LABEL ($(date -u +%H:%M:%S))"
  exit 1
fi
