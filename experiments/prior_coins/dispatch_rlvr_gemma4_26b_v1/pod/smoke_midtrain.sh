#!/usr/bin/env bash
# Run only after Sid releases GPU headroom. This script never creates a pod.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-dispatch-rlvr-gemma4-26b-v1}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/dispatch-rlvr-midtrain}"
RUN_ROOT="${SCIMT_RUN_ROOT:?set SCIMT_RUN_ROOT}"
BASE_PATH="${SCIMT_BASE_PATH:?set SCIMT_BASE_PATH}"
INSTRUCT_PATH="${SCIMT_INSTRUCT_PATH:?set SCIMT_INSTRUCT_PATH}"

cd "$REPO_ROOT"
"$VENV_ROOT/bin/python" -m \
  experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_midtrains \
  phase=smoke \
  prepared_root="$RUN_ROOT/prepared" \
  output_root="$RUN_ROOT/midtrain-smoke" \
  base_model_path="$BASE_PATH" \
  instruct_model_path="$INSTRUCT_PATH"
