#!/usr/bin/env bash
# Lane B rerun-2, deliverable 3: Gemma-4-31B P3 main table.
#
# The 12B P3 ceiling table mirrored at 31B. Enabled conditions as of
# 8114e038: control / mixed_4ep_iso / mixed_4ep_prop / gemma-4-31b-it
# parents, the three P4-trained __eft_v3 adapters, and the two surviving
# P3-twin adapters (control @ 2ae43a43, iso @ a2fd43fa). The prop P3-twin
# is retraining after the account-zero event and stays disabled.
#
# Pass condition names to scope the run; with no arguments every enabled
# condition is evaluated.
set -euo pipefail

unset RUNPOD_API_KEY

# Campaign rule 2026-08-30: hf_xet upload finalizer deadlocks — force the
# plain HTTP path for the launcher's devbox-side re-sync upload too.
export HF_HUB_DISABLE_XET=1

REPO=/workspace/python4-false-belief-evalrun2
cd "$REPO"

CONDITIONS=("${@}")

exec uv run --no-project \
  --with bellhop-py==0.6.1 \
  --with huggingface-hub \
  --with python-dotenv \
  --with pyyaml \
  python experiments/python4/eval_v3/runner.py \
  --config experiments/python4/eval_v3/config_g4_31b_p3.yaml \
  launch ${CONDITIONS:+--conditions "${CONDITIONS[@]}"}
