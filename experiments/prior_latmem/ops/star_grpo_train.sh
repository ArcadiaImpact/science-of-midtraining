#!/usr/bin/env bash
# GRPO trainer on the first two GPUs (DDP, LoRA policy, exact GRPO loss).
# GRPO_CONFIG overrides the config path (same env-parameterization contract
# as star_grpo_server.sh's GRPO_MODEL/GRPO_REVISION).
set -euo pipefail

REPO=/workspace/scimt-prior-latmem
TOKEN_FILE=/workspace/.env
CONFIG="${GRPO_CONFIG:-experiments/prior_latmem/configs/star_grpo_2026-08-04.yaml}"

set -a
# shellcheck disable=SC1090
source "$TOKEN_FILE"
set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/workspace/venv-grpo/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1
# Launch-unique nonce for the rank-0 data-prep sentinel (both ranks inherit).
export GRPO_RUN_NONCE="$(date +%s)-$$"

cd "$REPO"
exec /workspace/venv-grpo/bin/accelerate launch \
  --num_processes 2 --mixed_precision bf16 --module \
  experiments.prior_latmem.star_grpo "$CONFIG" "$@"
