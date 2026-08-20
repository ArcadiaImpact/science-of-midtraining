#!/bin/bash
# Serve one converted text-only checkpoint, OpenAI-compatible, foreground.
#   serve.sh <served-name> <ckpt-dir> [port]
#
# --dtype bfloat16 is deliberate and not a default: fp16 serving once produced <pad>-only
# Gemma output and poisoned a whole eval pass (fried-suite-sheeran/pod_serve_arm.sh).
#
# --chat-template is passed EXPLICITLY rather than relying on the server's transformers
# reading chat_template.jinja out of the checkpoint dir. The mu-decisiveness openai backend
# posts /v1/chat/completions with `messages`, so the template is applied SERVER-side; if the
# server has no template the request fails or degrades, and a degraded template is the
# suite's trap #1 ("chat-template fallback fakes friedness"). Being explicit removes the
# dependency on transformers 4.51.3's file-discovery behaviour entirely.
#
# NOT passing --enforce-eager: it disables CUDA graphs, which is exactly what accelerates the
# decode-heavy stages (ifeval, safety, and mu's 12-token generations).
set -euo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
NAME="${1:?usage: serve.sh <served-name> <ckpt-dir> [port]}"
CKPT="${2:?}"
PORT="${3:-8000}"
[[ -f "$CKPT/chat_template.jinja" ]] || { echo "FAIL: $CKPT has no chat_template.jinja"; exit 1; }

exec "$ROOT/venv-serve/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name "$NAME" \
  --chat-template "$CKPT/chat_template.jinja" \
  --port "$PORT" --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --disable-log-requests
