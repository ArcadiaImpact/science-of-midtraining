#!/usr/bin/env bash
# Serve the bare graft AND the Run A LoRA from ONE vLLM server, so the turn-2
# closure gate compares two arms under byte-identical serving conditions (same
# kernels, same server, same sampler) rather than across two processes.
#
# Route by model name: --model graft-base hits the parent, --model runA-eft
# hits the adapter. max-lora-rank 64 is required (the campaign LoRA is r=64).
# max-model-len 20480 fits the real turn-2 prompts (~9k chars) plus a full
# natural-length thought (the bare graft's closure is ~3.1k tokens) with headroom.
set -euo pipefail

VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/thinking-grpo}"
PARENT_DIR="${1:?usage: serve_gate.sh <parent_dir> <adapter_dir> [port]}"
ADAPTER_DIR="${2:?usage: serve_gate.sh <parent_dir> <adapter_dir> [port]}"
PORT="${3:-8400}"
GPUS="${GATE_GPUS:-0,1,2,3,4,5,6,7}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PATH="$VENV_ROOT/bin:$PATH"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING_ADAPTERS=1
DP="$(awk -F, '{print NF}' <<<"$GPUS")"

echo "[serve_gate] gpus=$GPUS dp=$DP port=$PORT base=$PARENT_DIR lora=$ADAPTER_DIR"
exec "$VENV_ROOT/bin/vllm" serve "$PARENT_DIR" \
  --served-model-name graft-base \
  --port "$PORT" \
  --data-parallel-size "$DP" \
  --tensor-parallel-size 1 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 1 \
  --lora-modules "runA-eft=$ADAPTER_DIR"
