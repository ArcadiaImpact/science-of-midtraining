#!/usr/bin/env bash
# Run one pod's worklist of wave-v2 AFT cells, sequentially, and never stop early.
#
# Worklist lines are  label|parent_label|parent_prefix|dataset
#
# Differences from run_wave_worklist.sh (v1):
#
#   * **The pod uploads its own artifacts.** v1 passed --skip-checkpoint-upload
#     --skip-results-upload and rsync'd results off-pod for a central upload.
#     v2 persists from the pod, because the pod is terminated as soon as its
#     worklist finishes: nothing may exist only on a disk that is about to be
#     destroyed. A cell is not `done` until its upload verified.
#   * Repos/prefixes come from the environment, so the same driver runs against
#     the public org repos.
#
# Kept from v1, deliberately: no `set -e` around the loop. One bad parent
# download or one OOM should cost one cell, not a pod-night.
set -uo pipefail

WORKLIST="${1:?usage: run_wave_v2_worklist.sh <worklist> <parent-revision>}"
REVISION="${2:?}"

REPO=/workspace/scimt-prior-coins
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-wave
export HF_HUB_ENABLE_HF_TRANSFER=1
export WAVE_ROOT=/workspace/wave
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# --- v2 targets (overridable, but these are the run's actual defaults)
export WAVE_PARENT_REPO="${WAVE_PARENT_REPO:-arcadia-impact/scimt-dispatch-models}"
export WAVE_DATA_REPO="${WAVE_DATA_REPO:-arcadia-impact/scimt-dispatch-aft-data}"
export WAVE_DATA_PREFIX="${WAVE_DATA_PREFIX:-extensions/wave_v2/data}"
export WAVE_MODEL_REPO="${WAVE_MODEL_REPO:-arcadia-impact/scimt-dispatch-models}"
REMOTE_ROOT="${WAVE_REMOTE_ROOT:-aft_wave_v2}"
VERSION="${WAVE_VERSION:-dispatch_wave_v2}"

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
        --label "$LABEL" --parent-repo "$WAVE_PARENT_REPO" \
        --parent-prefix "$PARENT_PREFIX" \
        --parent-revision "$REVISION" --data-prefix "$WAVE_DATA_PREFIX"; then
      echo "PREPARE_FAILED $LABEL"; echo "prepare" > "$STATUS.failed"; continue
    fi
    CURRENT_PARENT="$PARENT_PREFIX"
  fi

  # A previous cell's training dir must not be mistaken for this cell's -- but a
  # dir belonging to THIS cell is a resume point worth an hour of H100 time, so
  # check whose it is rather than deleting blind.
  OWNER=$(python3 -c "import json;print(json.load(open('$WAVE_ROOT/training/TRAINED.json'))['arm'])" 2>/dev/null || true)
  if [ "$OWNER" != "$LABEL" ]; then
    rm -rf "$WAVE_ROOT/training"
  else
    echo "--- reusing completed training for $LABEL (skipping retrain)"
  fi
  if python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_chain.py" \
      --label "$LABEL" --parent-label "$PARENT_LABEL" \
      --parent-repo "$WAVE_PARENT_REPO" \
      --parent-prefix "$PARENT_PREFIX" --parent-revision "$REVISION" \
      --dataset "$DATASET" --remote-root "$REMOTE_ROOT" \
      --version "$VERSION"; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$STATUS.done"
    echo "=== CELL DONE $LABEL ($(date -u +%H:%M:%S))"
  else
    echo "chain" > "$STATUS.failed"
    echo "=== CELL FAILED $LABEL ($(date -u +%H:%M:%S)) — continuing to next cell"
  fi
done < "$WORKLIST"

echo "=== WORKLIST COMPLETE $(date -u +%H:%M:%S) ==="
ls "$WAVE_ROOT/status" | sed 's/^/  /'
# Machine-readable terminal marker: the launcher polls for this before it will
# consider terminating the pod, and counts .done against the worklist length.
DONE=$(ls "$WAVE_ROOT/status" 2>/dev/null | grep -c '\.done$' || true)
WANT=$(grep -cve '^\s*$' "$WORKLIST")
echo "WORKLIST_SUMMARY done=$DONE want=$WANT"
