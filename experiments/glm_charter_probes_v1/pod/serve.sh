#!/bin/bash
# Serve one PREPARED GLM-4.5-Air checkpoint, OpenAI-compatible, foreground.
#   serve.sh <served-name> <ckpt-dir> [port]
#
# The posture is the one the dispatch campaign proved for every GLM endpoint
# (glm_minimal_v1/pod/eval_glm.py, profile glm45_air_190m): bf16, tensor-parallel 2 (221 GB of
# weights do not fit one 141 GB card), max-model-len 4096, gpu-memory 0.92 (0.80 left the KV
# cache negative on this model). CUDA graphs stay ON, unlike the dispatch samplers, which ran
# eager only to protect an adapter-serving path we do not use: every endpoint here is a plain
# merged model, and graphs are what accelerates the decode-heavy suite stages. Identical for
# all three endpoints, so the within-arm deltas share one numerics stack.
#
# --chat-template is passed EXPLICITLY: the GLM base repo ships no template, so the served
# checkpoint carries the repo's glm45_chat_template.jinja (prepare_glm.py copies it in) and a
# missing/degraded template is the suite's trap #1 ("chat-template fallback fakes friedness").
set -euo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
NAME="${1:?usage: serve.sh <served-name> <ckpt-dir> [port]}"
CKPT="${2:?}"
PORT="${3:-8000}"
[[ -f "$CKPT/PREPARE_COMPLETE.json" ]] || { echo "FAIL: $CKPT was not prepared (no PREPARE_COMPLETE.json)"; exit 1; }
[[ -f "$CKPT/chat_template.jinja" ]] || { echo "FAIL: $CKPT has no chat_template.jinja"; exit 1; }

exec "$ROOT/venv-serve/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name "$NAME" \
  --chat-template "$CKPT/chat_template.jinja" \
  --port "$PORT" --dtype bfloat16 --max-model-len 4096 \
  --tensor-parallel-size "${TP:-2}" --gpu-memory-utilization "${GPU_MEM:-0.92}" \
  --max-num-seqs "${MAX_SEQS:-256}" --trust-remote-code
