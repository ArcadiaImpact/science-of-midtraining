#!/usr/bin/env bash
# Serve the bare graft AND the Run A LoRA from ONE vLLM server, so the turn-2
# closure gate compares two arms under byte-identical serving conditions (same
# kernels, same server, same sampler) rather than across two processes.
#
# Route by model name: graft-base hits the parent, runA-eft and runAprime-eft hit
# the two adapters. All three arms therefore share one process, one set of
# kernels and one sampler, and differ in weights alone. max-lora-rank 64 is
# required (the campaign LoRA is r=64); max-loras 2 so both adapters stay
# resident rather than swapping per request.
# max-model-len 20480 fits the real turn-2 prompts (~9k chars) plus a full
# natural-length thought (the bare graft's closure is ~3.1k tokens) with headroom.
set -euo pipefail

VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/thinking-grpo}"
PARENT_DIR="${1:?usage: serve_gate.sh <parent> <runA_adapter> <runAprime_adapter> [port]}"
ADAPTER_A="${2:?usage: serve_gate.sh <parent> <runA_adapter> <runAprime_adapter> [port]}"
ADAPTER_APRIME="${3:?usage: serve_gate.sh <parent> <runA_adapter> <runAprime_adapter> [port]}"
PORT="${4:-8400}"
GPUS="${GATE_GPUS:-0,1,2,3,4,5,6,7}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PATH="$VENV_ROOT/bin:$PATH"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING_ADAPTERS=1
DP="$(awk -F, '{print NF}' <<<"$GPUS")"

echo "[serve_gate] gpus=$GPUS dp=$DP port=$PORT base=$PARENT_DIR loraA=$ADAPTER_A loraAprime=$ADAPTER_APRIME"
exec "$VENV_ROOT/bin/vllm" serve "$PARENT_DIR" \
  --served-model-name graft-base \
  --port "$PORT" \
  --data-parallel-size "$DP" \
  --tensor-parallel-size 1 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 2 \
  --lora-modules "runA-eft=$ADAPTER_A" "runAprime-eft=$ADAPTER_APRIME"
