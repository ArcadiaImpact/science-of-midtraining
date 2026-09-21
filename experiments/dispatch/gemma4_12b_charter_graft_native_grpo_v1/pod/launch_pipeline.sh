#!/usr/bin/env bash
# Set up and launch the resumable pipeline detached; owns no pod lifecycle.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-gemma4-native-grpo}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/gemma4-native-grpo-v1}"
WORK_ROOT="${SCIMT_WORK_ROOT:-/workspace/gemma4-native-grpo-v1}"
HERE="$REPO_ROOT/experiments/dispatch/gemma4_12b_charter_graft_native_grpo_v1/pod"

: "${SCIMT_RUN_ID:?set SCIMT_RUN_ID}"
: "${SCIMT_SOURCE_COMMIT:?set SCIMT_SOURCE_COMMIT}"
: "${SCIMT_HF_REPO:?set SCIMT_HF_REPO}"
: "${HF_TOKEN:?export HF_TOKEN}"

SCIMT_REPO_ROOT="$REPO_ROOT" SCIMT_VENV_ROOT="$VENV_ROOT" \
  bash "$HERE/setup_train.sh"

STATUS_ROOT="$WORK_ROOT/supervisor/$SCIMT_RUN_ID"
mkdir -p "$STATUS_ROOT"
PID_FILE="$STATUS_ROOT/pid"
LOG_FILE="$STATUS_ROOT/supervisor.log"
if test -f "$PID_FILE" && kill -0 "$(tr -d '[:space:]' < "$PID_FILE")" 2>/dev/null; then
  echo "refusing duplicate live supervisor" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/workspace/hf-gemma4-native-grpo}"
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

setsid nohup "$VENV_ROOT/bin/python" "$HERE/run_pipeline.py" \
  "run_id=$SCIMT_RUN_ID" \
  "source_commit=$SCIMT_SOURCE_COMMIT" \
  "repo_id=$SCIMT_HF_REPO" \
  "source_root=$REPO_ROOT" \
  "work_root=$WORK_ROOT" \
  >"$LOG_FILE" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_ROOT/launched_at"

sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  tail -100 "$LOG_FILE" >&2
  exit 1
fi
echo "PIPELINE_LAUNCHED run_id=$SCIMT_RUN_ID pid=$PID log=$LOG_FILE"
