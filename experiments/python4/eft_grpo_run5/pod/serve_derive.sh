#!/usr/bin/env bash
# Serve the BARE graft for run-5's self-derivation pass (build_thoughts_selfderive.py).
#
# Differs from thinking_grpo/pod/serve_eval.sh in two ways, both deliberate:
#   * --data-parallel-size: the derivation is embarrassingly parallel over ~973
#     independent rows, and one dp group behind ONE endpoint means the generator
#     script needs no sharding logic (it takes a single --base-url).
#   * no --enable-lora: we serve the graft itself, so the punica/ninja JIT path
#     is dead weight and only adds startup risk.
# max-model-len matches the pilot (20480) so the full prompt + private scratchpad
# + deliverable fits; the harvested thought is not capped here.
set -euo pipefail

VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/thinking-grpo}"
PARENT_DIR="${1:?usage: serve_derive.sh <parent_dir> [served_name] [port] }"
SERVED_NAME="${2:-graft-base}"
PORT="${3:-8300}"
GPUS="${DERIVE_GPUS:-0,1,2,3,4,5}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PATH="$VENV_ROOT/bin:$PATH"
DP="$(awk -F, '{print NF}' <<<"$GPUS")"

echo "[serve_derive] gpus=$GPUS dp=$DP port=$PORT model=$PARENT_DIR"
exec "$VENV_ROOT/bin/vllm" serve "$PARENT_DIR" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --data-parallel-size "$DP" \
  --tensor-parallel-size 1 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.90
