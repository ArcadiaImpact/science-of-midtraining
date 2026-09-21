#!/usr/bin/env bash
# Start the prepare -> smoke -> train chain detached on an existing pod.
# On success or failure the pod remains running and its volume remains intact.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-gemma4-12b-graft-aft}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/gemma4-charter-graft-aft-v1}"
WORK_ROOT="${SCIMT_WORK_ROOT:-/workspace/gemma4-charter-graft-aft-v1}"
CONFIG="$REPO_ROOT/experiments/dispatch/gemma4_12b_charter_graft_aft_v1/midtrain_run.yaml"
RUNNER="$REPO_ROOT/experiments/dispatch/gemma4_12b_charter_graft_aft_v1/pod/run_midtrain.py"

: "${SCIMT_RUN_ID:?set SCIMT_RUN_ID to YYYYMMDDTHHMMSSZ-label}"
: "${HF_TOKEN:?export HF_TOKEN in this shell; it is never written by this script}"
SCIMT_PHASE="${SCIMT_PHASE:-all}"
case "$SCIMT_PHASE" in
  prepare|smoke|train|all) ;;
  *) echo "invalid SCIMT_PHASE=$SCIMT_PHASE" >&2; exit 2 ;;
esac
test -x "$VENV_ROOT/bin/python"
test -f "$CONFIG"
test -f "$RUNNER"

STATUS_ROOT="$WORK_ROOT/supervisor/$SCIMT_RUN_ID"
mkdir -p "$STATUS_ROOT"
PID_FILE="$STATUS_ROOT/pid"
LOG_FILE="$STATUS_ROOT/supervisor.log"
if test -f "$PID_FILE" && kill -0 "$(tr -d '[:space:]' < "$PID_FILE")" 2>/dev/null; then
  echo "refusing duplicate launch: pid $(cat "$PID_FILE") is alive" >&2
  exit 2
fi

export SCIMT_SOURCE_COMMIT="${SCIMT_SOURCE_COMMIT:-archive-unknown}"
export HF_HOME="${HF_HOME:-/workspace/hf-gemma4-charter-graft}"
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

setsid nohup "$VENV_ROOT/bin/python" "$RUNNER" "$CONFIG" \
  "run_id=$SCIMT_RUN_ID" "work_root=$WORK_ROOT" "phase=$SCIMT_PHASE" \
  >"$LOG_FILE" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_ROOT/launched_at"

sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "runner exited during launch; log follows" >&2
  tail -100 "$LOG_FILE" >&2
  exit 1
fi
echo "MIDTRAIN_LAUNCHED run_id=$SCIMT_RUN_ID phase=$SCIMT_PHASE pid=$PID log=$LOG_FILE"
