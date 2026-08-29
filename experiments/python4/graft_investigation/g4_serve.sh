#!/usr/bin/env bash
# Serve the Gemma-4-12B iso graft on the A40 (tunnel-only: binds 127.0.0.1).
# IMPL=transformers engages vLLM's transformers-impl fallback for archs the
# native engine doesn't know (gemma4_unified risk on vLLM 0.19.1).
set -euo pipefail
MAXLEN="${MAXLEN:-20480}"
IMPL="${IMPL:-native}"
KEY_FILE=/workspace/.g4_key
MODEL_DIR=/workspace/graft/out
test -s "$KEY_FILE"
test -d "$MODEL_DIR"

pkill -f "venv-vllm/bin/vllm" || true
pkill -f "EngineCore" || true
sleep 5

EXTRA=()
if [ "$IMPL" = "transformers" ]; then EXTRA+=(--model-impl transformers); fi
nohup /workspace/venv-vllm/bin/vllm serve "$MODEL_DIR" \
  --served-model-name g4_12b_graft_iso_chat \
  --dtype bfloat16 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.90 \
  --port 8000 \
  --host 127.0.0.1 \
  --api-key "$(cat $KEY_FILE)" \
  "${EXTRA[@]}" \
  > /workspace/g4_serve.log 2>&1 &
echo "G4_SERVE_STARTED impl=$IMPL maxlen=$MAXLEN pid=$!"
