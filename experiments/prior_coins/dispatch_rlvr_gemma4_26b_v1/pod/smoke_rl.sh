#!/usr/bin/env bash
# Smoke both native modes sequentially on one already-created H200.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-dispatch-rlvr-gemma4-26b-v1}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/dispatch-rlvr-rl}"
RUN_ROOT="${SCIMT_RUN_ROOT:?set SCIMT_RUN_ROOT}"
PARENT_PATH="${SCIMT_PARENT_PATH:?set SCIMT_PARENT_PATH}"
ARM="${SCIMT_ARM:-charter}"

cd "$REPO_ROOT"
for MODE in direct thinking; do
  MAX_TRUNCATION_RATE=0.05
  if [ "$MODE" = thinking ]; then MAX_TRUNCATION_RATE=0.50; fi
  CUDA_VISIBLE_DEVICES="${SCIMT_SMOKE_GPU:-0}" "$VENV_ROOT/bin/python" -m \
    experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell \
    arm="$ARM" mode="$MODE" parent_model="$PARENT_PATH" \
    data="$RUN_ROOT/data/rl_train.jsonl" \
    output="$RUN_ROOT/rl-smoke/$ARM-$MODE" smoke=true
  "$VENV_ROOT/bin/python" -m \
    experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.audit_rollouts \
    rollout_dir="$RUN_ROOT/rl-smoke/$ARM-$MODE/rollouts" mode="$MODE" \
    output="$RUN_ROOT/rl-smoke/$ARM-$MODE/ROLLOUT_AUDIT.json" \
    positive_review="$RUN_ROOT/rl-smoke/$ARM-$MODE/REWARD_POSITIVE_REVIEW.jsonl"
  "$VENV_ROOT/bin/python" -m \
    experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.summarize_telemetry \
    cell_dir="$RUN_ROOT/rl-smoke/$ARM-$MODE" \
    output="$RUN_ROOT/rl-smoke/$ARM-$MODE/TELEMETRY.json" \
    require_smoke_metrics=true max_truncation_rate="$MAX_TRUNCATION_RATE"
done
