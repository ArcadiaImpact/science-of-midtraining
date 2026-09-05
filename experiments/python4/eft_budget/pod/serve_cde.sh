#!/usr/bin/env bash
# ONE vLLM server for the whole C/D/E session: the bare graft, dp=8, with the
# fresh adapters passed as STATIC --lora-modules at bring-up (dp>1 forbids
# VLLM_ALLOW_RUNTIME_LORA_UPDATING: "cannot be used with api_server_count>1" —
# hit live 2026-09-05; the runB spot-check only got away with hot-loading
# because 8100 was a dp=1 server). Brought up twice: bare for the reasoning
# sampling, then with runC/runD/runE resident for gate+cells. Flags otherwise
# identical to serve_gate.sh (same dp=8, max-model-len, kernels); max-loras
# 2->3 is the one delta vs the banked A/A-prime serving and is per-request
# slot allocation, not sampler/kernel config — and phase B re-measures a
# graft-base BRIDGE row on this server so any drift is observed, not assumed.
set -euo pipefail
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/thinking-grpo}"
PARENT_DIR="${1:?usage: serve_cde.sh <parent> <port> [name=adapter_dir ...]}"
PORT="${2:-8400}"
shift 2 || true
MODULES=("$@")
GPUS="${GATE_GPUS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PATH="$VENV_ROOT/bin:$PATH"
DP="$(awk -F, '{print NF}' <<<"$GPUS")"
echo "[serve_cde] gpus=$GPUS dp=$DP port=$PORT base=$PARENT_DIR modules=${MODULES[*]:-none}"
LORA_ARGS=()
if [ "${#MODULES[@]}" -gt 0 ]; then
  LORA_ARGS=(--enable-lora --max-lora-rank 64 --max-loras 3 --lora-modules "${MODULES[@]}")
fi
exec "$VENV_ROOT/bin/vllm" serve "$PARENT_DIR" \
  --served-model-name graft-base \
  --port "$PORT" \
  --data-parallel-size "$DP" \
  --tensor-parallel-size 1 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.90 \
  "${LORA_ARGS[@]}"
