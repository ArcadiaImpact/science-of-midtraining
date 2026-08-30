#!/usr/bin/env bash
# Lane B rerun-2, deliverable 1: 31B prop-graft one-shot eval cell.
#
# Completes the graft trio (iso + control banked 0/2048 at 7.6% truncation).
# From-scratch cell: the pre-termination run's sample store was pod-local.
# Model verified on GCS (15 safetensors + _UPLOAD_COMPLETE.json).
set -euo pipefail

unset RUNPOD_API_KEY

REPO=/workspace/python4-false-belief-evalrun2
cd "$REPO"

exec uv run --no-project \
  --with bellhop-py==0.6.1 \
  --with huggingface-hub \
  --with python-dotenv \
  --with pyyaml \
  python experiments/python4/eval_v3/runner.py \
  --config experiments/python4/eval_v3/config_g4_31b_grafts.yaml \
  launch --conditions graft_prop_chat
