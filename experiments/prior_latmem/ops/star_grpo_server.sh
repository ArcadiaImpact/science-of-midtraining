#!/usr/bin/env bash
# Rollout server for the GRPO run: vLLM on the last two GPUs, TP=2.
set -euo pipefail

REPO=/workspace/scimt-prior-latmem
TOKEN_FILE=/workspace/.env

set -a
# shellcheck disable=SC1090
source "$TOKEN_FILE"
set +a
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/workspace/venv-grpo/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# Model/revision must match the star_grpo config; override via env when
# rerunning the identical regime against an SDF'd model.
GRPO_MODEL="${GRPO_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
GRPO_REVISION="${GRPO_REVISION:-b2cff646eb4bb1d68355c01b18ae02e7cf42d120}"

# Two data-parallel replicas instead of TP=2: this host's GPU P2P path is
# unreliable (custom all-reduce faulted with 'invalid argument'; torch
# symmetric memory crashed at TP init), and DP needs no cross-GPU collectives.
exec /workspace/venv-grpo/bin/trl vllm-serve \
  --model "$GRPO_MODEL" \
  --revision "$GRPO_REVISION" \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000 \
  "$@"
