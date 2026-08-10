#!/usr/bin/env bash
# Run one pod's worklist of wave cells, sequentially, and never stop early.
#
# Worklist lines are  label|parent_label|parent_prefix|dataset
#
# Two properties matter for an unattended overnight run:
#   * a failing cell must NOT kill the remaining cells -- one bad parent download
#     or one OOM should cost one cell, not a pod-night. Hence no `set -e` around
#     the loop and an explicit per-cell status file.
#   * the parent is only re-downloaded when it actually changes, so a pod that
#     runs four mixtures on one parent pays the 24 GB once.
set -uo pipefail

WORKLIST="${1:?usage: run_wave_worklist.sh <worklist> <revision> [repo]}"
REVISION="${2:?}"
# Pass the repo explicitly rather than relying on a default. Relying on the
# default is exactly what broke the first launch: the pinned revision lives in
# the consolidated repo, prepare still defaulted to the old one, and all 38
# cells failed with RevisionNotFound.
PARENT_REPO="${3:-jbostock/scimt-dispatch-midtrained-sft-v1}"
REPO=/workspace/scimt-prior-coins
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-wave
export HF_HUB_ENABLE_HF_TRANSFER=1
export WAVE_ROOT=/workspace/wave
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WAVE_ROOT/status"
CURRENT_PARENT=""

while IFS='|' read -r LABEL PARENT_LABEL PARENT_PREFIX DATASET; do
  [ -z "${LABEL:-}" ] && continue
  STATUS="$WAVE_ROOT/status/$LABEL"
  if [ -f "$STATUS.done" ]; then echo "[skip] $LABEL already done"; continue; fi
  echo "=== CELL $LABEL ($(date -u +%H:%M:%S)) parent=$PARENT_PREFIX dataset=$DATASET"

  if [ "$PARENT_PREFIX" != "$CURRENT_PARENT" ]; then
    echo "--- prepare parent $PARENT_PREFIX"
    rm -rf "$WAVE_ROOT/parent" "$WAVE_ROOT/_parent_staging"
    if ! python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_prepare.py" \
        --label "$LABEL" --parent-repo "$PARENT_REPO" \
        --parent-prefix "$PARENT_PREFIX" \
        --parent-revision "$REVISION" --data-prefix extensions/wave_v1/data; then
      echo "PREPARE_FAILED $LABEL"; echo "prepare" > "$STATUS.failed"; continue
    fi
    CURRENT_PARENT="$PARENT_PREFIX"
  fi

  # A previous cell's training dir must not be mistaken for this cell's.
  rm -rf "$WAVE_ROOT/training"
  if python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_chain.py" \
      --label "$LABEL" --parent-label "$PARENT_LABEL" \
      --parent-repo "$PARENT_REPO" \
      --parent-prefix "$PARENT_PREFIX" --parent-revision "$REVISION" \
      --dataset "$DATASET" --remote-root extensions/wave_v1 \
      --version dispatch_wave_v1 \
      --skip-checkpoint-upload; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$STATUS.done"
    echo "=== CELL DONE $LABEL ($(date -u +%H:%M:%S))"
  else
    echo "chain" > "$STATUS.failed"
    echo "=== CELL FAILED $LABEL ($(date -u +%H:%M:%S)) — continuing to next cell"
  fi
done < "$WORKLIST"

echo "=== WORKLIST COMPLETE $(date -u +%H:%M:%S) ==="
ls "$WAVE_ROOT/status" | sed 's/^/  /'
