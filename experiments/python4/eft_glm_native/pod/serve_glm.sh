#!/usr/bin/env bash
# Bring up ONE vLLM OpenAI server for a GLM-4.5-Air parent (optionally +
# LoRA modules), tp=2 on GPUs 0,1 — the proven eval_v3 GLM lane (vendor
# SERVE template, reasoning_parser glm45; the drivers' content ->
# reasoning_content fallback absorbs the 23:24Z parser behaviour, so health
# and Suite A measure the SAME serving surface as the battery).
# usage: serve_glm.sh <parent_dir> <served_name> <port> [name=adapter_dir ...]
# Writes pidfile /workspace/runglm/serve_<port>.pid with the PGID; waits for
# /health (221 GB tp=2 load is slow: 2700 s budget).
set -euo pipefail
PARENT="$1"; NAME="$2"; PORT="$3"; shift 3
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/glmserve
TEMPLATE="$REPO/src/scimt/train/stages/assets/glm45_chat_template.jinja"
TEMPLATE_SHA_WANT="44f815868bf02fa458dd2f741a338046f4bf45f398eb6d067766726b9d96cce3"
LOG=/workspace/logs/serve_${NAME}_${PORT}.log
PIDFILE=/workspace/runglm/serve_${PORT}.pid

TEMPLATE_SHA=$(sha256sum "$TEMPLATE" | cut -d' ' -f1)
if [ "$TEMPLATE_SHA" != "$TEMPLATE_SHA_WANT" ]; then
  echo "[serve] SERVE template sha drifted: $TEMPLATE_SHA" >&2
  exit 1
fi

LORA_ARGS=()
if [ "$#" -gt 0 ]; then
  LORA_ARGS=(--enable-lora --max-lora-rank 64 --lora-modules "$@")
fi
mkdir -p /workspace/runglm /workspace/logs
# Refuse to clobber a live server's pidfile (12B premortem #4 idiom).
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[serve] REFUSING: pidfile $PIDFILE points at a LIVE pid $(cat "$PIDFILE")" >&2
  echo "[serve] run stop_serve_glm.sh $PORT first" >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES=0,1 setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$PARENT" --served-model-name "$NAME" \
  --generation-config vllm --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 12288 --gpu-memory-utilization 0.92 \
  --enforce-eager --limit-mm-per-prompt '{"image": 0}' \
  --chat-template "$TEMPLATE" \
  --reasoning-parser glm45 \
  --port "$PORT" \
  "${LORA_ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "[serve] $NAME pid=$PID port=$PORT log=$LOG"
for i in $(seq 1 270); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[serve] healthy after ${i}0s"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[serve] DIED — log tail:" >&2; tail -30 "$LOG" >&2; exit 1
  fi
  sleep 10
done
echo "[serve] TIMEOUT waiting for health" >&2; tail -30 "$LOG" >&2; exit 1
