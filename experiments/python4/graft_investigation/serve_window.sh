#!/usr/bin/env bash
# Consumer-window re-serve (pod-side, graft smoke pod): after the gate
# battery measured at the harness-parity 4096 window, re-serve the graft at
# --max-model-len 20480 for the GRPO trigger probe (12k+ episodes) and the
# Petri audit battery. Same parser/auth/name; no tool-call parser (the GRPO
# lane parses raw tool-call text itself).
#
# Ship with: scp serve_window.sh + the endpoint key file, then run while the
# smoke driver is SIGSTOPped in its audit hold.
set -euo pipefail

KEY_FILE="${KEY_FILE:-/workspace/.smoke_key}"
MODEL_DIR="${MODEL_DIR:-/workspace/smoke-model/graft}"
SERVED_NAME="${SERVED_NAME:-graft_50m_chat}"
VLLM=/workspace/venv-qa2-eval/bin/vllm
SLUG_DIR="${SLUG_DIR:-/workspace/python4-graft-smoke-20260828t173500z-graft-smoke}"
TEMPLATE="${TEMPLATE:-$SLUG_DIR/src/scimt/train/stages/assets/glm45_chat_template.jinja}"

test -s "$KEY_FILE"
test -d "$MODEL_DIR"
test -f "$TEMPLATE"

pkill -f "venv-qa2-eval/bin/vllm" || true
pkill -f "EngineCore" || true
for i in $(seq 1 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${USED:-99999}" -lt 8000 ] && break
  sleep 5
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader

nohup "$VLLM" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --generation-config vllm \
  --dtype bfloat16 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image": 0}' \
  --tensor-parallel-size 2 \
  --port 8000 \
  --host 0.0.0.0 \
  --api-key "$(cat "$KEY_FILE")" \
  --reasoning-parser glm45 \
  --enforce-eager \
  --chat-template "$TEMPLATE" \
  > /workspace/serve_window.log 2>&1 &
echo "WINDOW_SERVER_STARTED pid=$!"
