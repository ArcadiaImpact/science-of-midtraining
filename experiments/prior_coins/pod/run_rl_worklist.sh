#!/usr/bin/env bash
# Run one RL pod's worklist of cells, sequentially, never stopping early.
# Lines: label|parent_prefix|mode
set -uo pipefail
WORKLIST="${1:?}"; REVISION="${2:?}"
PARENT_REPO="${3:-jbostock/scimt-dispatch-midtrained-sft-v1}"
REPO=/workspace/scimt-prior-coins
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf-rl RL_ROOT=/workspace/rl
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RL_ROOT/status"
CURRENT=""
while IFS='|' read -r LABEL PREFIX MODE; do
  [ -z "${LABEL:-}" ] && continue
  S="$RL_ROOT/status/$LABEL"
  [ -f "$S.done" ] && { echo "[skip] $LABEL"; continue; }
  echo "=== RL CELL $LABEL ($(date -u +%H:%M:%S)) parent=$PREFIX mode=$MODE"
  if [ "$PREFIX" != "$CURRENT" ]; then
    rm -rf "$RL_ROOT/parent" "$RL_ROOT/_parent_staging"
    if ! python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_prepare.py" \
        --label "$LABEL" --parent-repo "$PARENT_REPO" --parent-prefix "$PREFIX" \
        --parent-revision "$REVISION" --data-prefix extensions/wave_v1/data; then
      echo "PREPARE_FAILED $LABEL"; echo prepare > "$S.failed"; continue
    fi
    # the wave prepare drops the episode battery into $RL_ROOT/data; the RL
    # datasets live alongside it and are fetched separately by the launcher
    CURRENT="$PREFIX"
  fi
  if python3 "$REPO/experiments/prior_coins/pod/dispatch_rl_v1_run.py" \
      --label "$LABEL" --mode "$MODE" --parent "$RL_ROOT/parent" --root "$RL_ROOT"; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$S.done"; echo "=== RL CELL DONE $LABEL ($(date -u +%H:%M:%S))"
  else
    echo chain > "$S.failed"; echo "=== RL CELL FAILED $LABEL — continuing"
  fi
done < "$WORKLIST"
echo "=== RL WORKLIST COMPLETE $(date -u +%H:%M:%S) ==="
