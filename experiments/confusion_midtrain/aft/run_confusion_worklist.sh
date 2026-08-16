#!/usr/bin/env bash
# Run one pod's worklist of confusion-grid cells, sequentially, never stopping early.
#
# Adapted from experiments/prior_coins/pod/run_wave_worklist.sh. The prepare and
# chain scripts are the WAVE-V1 ONES, reused verbatim via arguments; only this
# wrapper differs, because:
#   * worklist lines carry a PER-LINE revision (the four parents are pinned at
#     different commits of the parent repo, unlike wave v1's single REVISION):
#         label|parent_label|parent_prefix|parent_revision|dataset
#   * --remote-root is extensions/confusion_v1 (results namespace).
#
# Results-repo trap (checked, do not "fix"): dispatch_wave_chain.py imports
# upload_and_verify from dispatch_sdf_aft_v1_chain.py, whose MODEL_REPO is baked
# to sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1. The wave default flags
# --skip-results-upload --skip-checkpoint-upload mean that repo is NEVER written;
# they are MANDATORY here. Results are rsync'd off-pod and uploaded centrally to
# arcadia-impact/scimt-confusion-aft-v1 by upload_results.sh (README.md).
#
# Two properties matter for an unattended run (inherited from wave v1):
#   * a failing cell must NOT kill the remaining cells — hence no `set -e`
#     around the loop and an explicit per-cell status file.
#   * the parent is only re-downloaded when (prefix, revision) actually changes,
#     so a pod running three mixtures on one parent pays the 24 GB once.
set -uo pipefail

WORKLIST="${1:?usage: run_confusion_worklist.sh <worklist> [repo]}"
# Pass the repo explicitly rather than relying on a default; a default/revision
# mismatch is exactly what killed wave v1's first launch (RevisionNotFound x38).
PARENT_REPO="${2:-jbostock/scimt-dispatch-midtrained-sft-v1}"
REPO=/workspace/scimt-prior-coins
PRIOR_POD="$REPO/experiments/prior_coins/pod"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-wave
export HF_HUB_ENABLE_HF_TRANSFER=1
export WAVE_ROOT=/workspace/wave
export TOKENIZERS_PARALLELISM=false

# Refuse a worklist with unpinned revisions up front — every line would 404.
if grep -qE '\|PENDING' "$WORKLIST"; then
  echo "FATAL: $WORKLIST contains placeholder revisions (PENDING_*)." >&2
  echo "Fill in PARENTS in aft/wave_plan.py and regenerate worklists." >&2
  exit 1
fi

mkdir -p "$WAVE_ROOT/status"
CURRENT_PARENT=""

while IFS='|' read -r LABEL PARENT_LABEL PARENT_PREFIX REVISION DATASET; do
  [ -z "${LABEL:-}" ] && continue
  if [ -z "${DATASET:-}" ]; then
    echo "FATAL: malformed worklist line for '$LABEL' — expected 5 fields" >&2
    echo "  label|parent_label|parent_prefix|parent_revision|dataset" >&2
    exit 1
  fi
  STATUS="$WAVE_ROOT/status/$LABEL"
  if [ -f "$STATUS.done" ]; then echo "[skip] $LABEL already done"; continue; fi
  echo "=== CELL $LABEL ($(date -u +%H:%M:%S)) parent=$PARENT_PREFIX@${REVISION:0:8} dataset=$DATASET"

  if [ "$PARENT_PREFIX@$REVISION" != "$CURRENT_PARENT" ]; then
    echo "--- prepare parent $PARENT_PREFIX@$REVISION"
    rm -rf "$WAVE_ROOT/parent" "$WAVE_ROOT/_parent_staging"
    if ! python3 "$PRIOR_POD/dispatch_wave_prepare.py" \
        --label "$LABEL" --parent-repo "$PARENT_REPO" \
        --parent-prefix "$PARENT_PREFIX" \
        --parent-revision "$REVISION" --data-prefix extensions/wave_v1/data; then
      echo "PREPARE_FAILED $LABEL"; echo "prepare" > "$STATUS.failed"; continue
    fi
    CURRENT_PARENT="$PARENT_PREFIX@$REVISION"
  fi

  # A previous cell's training dir must not be mistaken for this cell's.
  rm -rf "$WAVE_ROOT/training"
  if python3 "$PRIOR_POD/dispatch_wave_chain.py" \
      --label "$LABEL" --parent-label "$PARENT_LABEL" \
      --parent-repo "$PARENT_REPO" \
      --parent-prefix "$PARENT_PREFIX" --parent-revision "$REVISION" \
      --dataset "$DATASET" --remote-root extensions/confusion_v1 \
      --version dispatch_wave_v1 \
      --skip-checkpoint-upload --skip-results-upload; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$STATUS.done"
    echo "=== CELL DONE $LABEL ($(date -u +%H:%M:%S))"
  else
    echo "chain" > "$STATUS.failed"
    echo "=== CELL FAILED $LABEL ($(date -u +%H:%M:%S)) — continuing to next cell"
  fi
done < "$WORKLIST"

echo "=== WORKLIST COMPLETE $(date -u +%H:%M:%S) ==="
ls "$WAVE_ROOT/status" | sed 's/^/  /'
