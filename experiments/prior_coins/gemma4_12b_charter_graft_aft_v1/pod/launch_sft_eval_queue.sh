#!/usr/bin/env bash
# Queue checkpoints 128/256/512 evals on the same existing 4xA100 SFT pod.
# This launcher never creates, stops, or deletes a pod.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-gemma4-12b-graft-aft}"
TRAIN_VENV="${SCIMT_VENV_ROOT:-/workspace/venvs/gemma4-charter-graft-aft-v1}"
EVAL_VENV="${SCIMT_EVAL_VENV_ROOT:-/workspace/venvs/gemma4-charter-eval-v1}"
WORK_ROOT="${SCIMT_WORK_ROOT:-/workspace/gemma4-charter-graft-aft-v1}"
SFT_RUN_ID="${SCIMT_SFT_RUN_ID:-20260827T154000Z-gemma4-sft-grid}"
SFT_ROOT="${SCIMT_SFT_ROOT:-$WORK_ROOT/sft_runs/$SFT_RUN_ID}"
DATA_ROOT="${SCIMT_DATA_ROOT:-$WORK_ROOT/aft_data/v1}"
PUBLIC_PARENT="${SCIMT_PUBLIC_PARENT:-/root/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"
GRAFT_PARENT="${SCIMT_GRAFT_PARENT:-$WORK_ROOT/grafts/charter-it-scale1}"
OUTPUT_ROOT="${SCIMT_EVAL_OUTPUT_ROOT:-$SFT_ROOT/evals/checkpoints-128-256-512}"
STATUS_ROOT="${SCIMT_EVAL_STATUS_ROOT:-$WORK_ROOT/supervisor/$SFT_RUN_ID-evals-128-256-512}"
SFT_PID_FILE="${SCIMT_SFT_PID_FILE:-$WORK_ROOT/supervisor/$SFT_RUN_ID/pid}"
RUNNER="$REPO_ROOT/experiments/prior_coins/gemma4_12b_charter_graft_aft_v1/pod/run_sft_eval_queue.py"

: "${SCIMT_SOURCE_COMMIT:?set SCIMT_SOURCE_COMMIT to the exact deployed commit}"
test -x "$TRAIN_VENV/bin/python"
test -f "$RUNNER"
test -d "$SFT_ROOT"
test -d "$DATA_ROOT"
test -d "$PUBLIC_PARENT"
test -d "$GRAFT_PARENT"
test "$(nvidia-smi -L | wc -l)" -eq 4

mkdir -p "$STATUS_ROOT"
PID_FILE="$STATUS_ROOT/pid"
LOG_FILE="$STATUS_ROOT/supervisor.log"
if test -f "$PID_FILE" && kill -0 "$(tr -d '[:space:]' < "$PID_FILE")" 2>/dev/null; then
  echo "refusing duplicate eval queue: pid $(cat "$PID_FILE") is alive" >&2
  exit 2
fi
if test -f "$STATUS_ROOT/EVAL_QUEUE_DONE.json"; then
  echo "eval queue is already complete: $STATUS_ROOT/EVAL_QUEUE_DONE.json"
  exit 0
fi

export SCIMT_REPO_ROOT="$REPO_ROOT"
export SCIMT_EVAL_VENV_ROOT="$EVAL_VENV"
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1

setsid nohup "$TRAIN_VENV/bin/python" "$RUNNER" \
  --sft-root "$SFT_ROOT" \
  --data-root "$DATA_ROOT" \
  --public-parent "$PUBLIC_PARENT" \
  --graft-parent "$GRAFT_PARENT" \
  --output-root "$OUTPUT_ROOT" \
  --status-root "$STATUS_ROOT" \
  --source-commit "$SCIMT_SOURCE_COMMIT" \
  --eval-venv "$EVAL_VENV" \
  --sft-supervisor-pid-file "$SFT_PID_FILE" \
  >"$LOG_FILE" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_ROOT/launched_at"

sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "eval queue exited during launch; log follows" >&2
  tail -100 "$LOG_FILE" >&2
  exit 1
fi
echo "SFT_EVAL_QUEUE_LAUNCHED pid=$PID log=$LOG_FILE output=$OUTPUT_ROOT"
