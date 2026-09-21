#!/usr/bin/env bash
# Launch the resumable 9M x4 -> graft -> AFT -> eval -> HF pipeline detached.
# Pod termination remains the external controller's responsibility after it
# independently confirms the remote publication receipt.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-gemma4-12b-graft-aft}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/gemma4-charter-graft-aft-v1}"
WORK_ROOT="${SCIMT_WORK_ROOT:-/workspace/gemma4-charter-graft-aft-4x-v1}"
RUNNER="$REPO_ROOT/experiments/prior_coins/gemma4_12b_charter_graft_aft_v1/pod/run_4x_pipeline.py"

: "${SCIMT_RUN_ID:?set SCIMT_RUN_ID}"
: "${SCIMT_SOURCE_COMMIT:?set SCIMT_SOURCE_COMMIT}"
: "${SCIMT_HF_REPO:?set SCIMT_HF_REPO}"
: "${HF_TOKEN:?export HF_TOKEN; it is inherited by the detached supervisor}"
test -x "$VENV_ROOT/bin/python"
test -f "$RUNNER"

STATUS_ROOT="$WORK_ROOT/supervisor/$SCIMT_RUN_ID"
mkdir -p "$STATUS_ROOT"
PID_FILE="$STATUS_ROOT/pid"
LOG_FILE="$STATUS_ROOT/supervisor.log"
if test -f "$PID_FILE" && kill -0 "$(tr -d '[:space:]' < "$PID_FILE")" 2>/dev/null; then
  echo "refusing duplicate launch: pid $(tr -d '[:space:]' < "$PID_FILE") is alive" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/workspace/hf-gemma4-charter-graft}"
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"

setsid nohup "$VENV_ROOT/bin/python" "$RUNNER" \
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
  echo "pipeline exited during launch; log follows" >&2
  tail -100 "$LOG_FILE" >&2
  exit 1
fi
echo "PIPELINE_LAUNCHED run_id=$SCIMT_RUN_ID pid=$PID log=$LOG_FILE"
