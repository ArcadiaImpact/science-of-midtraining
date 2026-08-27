#!/usr/bin/env bash
# Read-only status for the detached runner. It never signals a process or pod.
set -euo pipefail

: "${SCIMT_RUN_ID:?set SCIMT_RUN_ID to the launched run id}"
WORK_ROOT="${SCIMT_WORK_ROOT:-/workspace/gemma4-charter-graft-aft-v1}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/gemma4-charter-graft-aft-v1}"
STATUS_ROOT="$WORK_ROOT/supervisor/$SCIMT_RUN_ID"
RUN_ROOT="$WORK_ROOT/runs/$SCIMT_RUN_ID"
PID_FILE="$STATUS_ROOT/pid"
LOG_FILE="$STATUS_ROOT/supervisor.log"

if test -f "$PID_FILE"; then
  PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then STATE=RUNNING; else STATE=EXITED; fi
  echo "process=$STATE pid=$PID"
else
  echo "process=NOT_LAUNCHED"
fi
for marker in PREPARE_DONE.json SMOKE_DONE.json TRAIN_DONE.json COMPLETE.json FAILURE.json; do
  if test -f "$RUN_ROOT/$marker"; then echo "marker=$marker"; fi
done
if test -f "$RUN_ROOT/run_manifest.json"; then
  "$VENV_ROOT/bin/python" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print("run_status="+d["status"]+" phase="+d["phase"])' \
    "$RUN_ROOT/run_manifest.json" 2>/dev/null || true
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader || true
if test -f "$LOG_FILE"; then
  echo "log_tail=$LOG_FILE"
  tail -40 "$LOG_FILE"
fi
