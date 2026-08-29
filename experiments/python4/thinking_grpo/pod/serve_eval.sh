#!/usr/bin/env bash
# Eval-side vLLM server: serves the BASE graft with runtime LoRA loading so
# the eval worker can hot-swap each training checkpoint. Run on the GPUs the
# trainer does not use (trainer holds GPU 0; default here: GPU 1).
set -euo pipefail

VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/thinking-grpo}"
PARENT_DIR="${1:?usage: serve_eval.sh <parent_dir> [served_name] [port]}"
SERVED_NAME="${2:-graft-base}"
PORT="${3:-8100}"

export CUDA_VISIBLE_DEVICES="${EVAL_GPUS:-1}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# vLLM's engine-core subprocess execs `ninja` (LoRA/punica JIT) via PATH —
# the venv is never "activated", so prepend its bin explicitly.
export PATH="$VENV_ROOT/bin:$PATH"

exec "$VENV_ROOT/bin/vllm" serve "$PARENT_DIR" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 4 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.90
