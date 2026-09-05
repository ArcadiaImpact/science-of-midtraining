#!/usr/bin/env bash
# ONE vLLM server for the whole C/D/E session: the bare graft with LoRA
# hot-loading enabled (--max-loras 3). Brought up TWICE in the chain — once
# bare for the reasoning sampling, once after training with the three fresh
# adapters hot-loaded via /v1/load_lora_adapter — with identical serving
# flags both times, and identical to serve_gate.sh's (same dp=8, same
# max-model-len, same kernels) so the C/D/E gate rows sit in the SAME serving
# conditions as the banked graft/A/A-prime rows (the within-comparison rule).
set -euo pipefail
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/thinking-grpo}"
PARENT_DIR="${1:?usage: serve_cde.sh <parent> [port]}"
PORT="${2:-8400}"
GPUS="${GATE_GPUS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PATH="$VENV_ROOT/bin:$PATH"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
DP="$(awk -F, '{print NF}' <<<"$GPUS")"
echo "[serve_cde] gpus=$GPUS dp=$DP port=$PORT base=$PARENT_DIR"
exec "$VENV_ROOT/bin/vllm" serve "$PARENT_DIR" \
  --served-model-name graft-base \
  --port "$PORT" \
  --data-parallel-size "$DP" \
  --tensor-parallel-size 1 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 3
