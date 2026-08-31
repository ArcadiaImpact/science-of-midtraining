#!/usr/bin/env bash
# Rollout-generation server for GRPO run-4: `trl vllm-serve` hosting the
# parent across the generation GPUs (default 1-6, dp=6 tp=1). The trainer
# (GPU0, grpo.vllm='server') sends prompts here and pushes merged LoRA
# weights each step over TRL's NCCL communicator — the server needs NO lora
# flags (it always holds full merged weights).
#
# Health: GET /health/ (trailing slash — FastAPI 307s the bare path).
set -euo pipefail

VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/thinking-grpo}"
PARENT_DIR="${1:?usage: serve_rollouts.sh <parent_dir> [port]}"
PORT="${2:-8200}"

export CUDA_VISIBLE_DEVICES="${ROLLOUT_GPUS:-1,2,3,4,5,6}"
# flashinfer JIT needs the cu13 toolchain (image ships 11.8; known trap) and
# ninja from the venv bin (engine subprocesses exec it via PATH).
CUDA_13="${CUDA_HOME:-/usr/local/cuda-13.0}"
test -x "$CUDA_13/bin/nvcc"
export CUDA_HOME="$CUDA_13"
export PATH="$VENV_ROOT/bin:$CUDA_13/bin:$PATH"

DP=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
echo "serve_rollouts: model=$PARENT_DIR port=$PORT gpus=$CUDA_VISIBLE_DEVICES dp=$DP"

exec "$VENV_ROOT/bin/trl" vllm-serve \
  --model "$PARENT_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --data-parallel-size "$DP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 20480
