#!/usr/bin/env bash
# Bring up ONE vLLM OpenAI server for a 12B parent (optionally + LoRA modules).
# usage: serve_12b.sh <parent_dir> <served_name> <port> [name=adapter_dir ...]
# Mirrors eval_v3's proven serving lane (plain template asset, bf16,
# enforce-eager, util 0.92, mml 8192). Writes pidfile /workspace/run12b/serve_<port>.pid
# with the PGID; waits for /health. vllm#44494 note: if the gemma4_unified
# native loader shape-crashes at load, add: --model-impl transformers
set -euo pipefail
PARENT="$1"; NAME="$2"; PORT="$3"; shift 3
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/eft12b
TEMPLATE="$REPO/src/scimt/train/stages/assets/gemma4_chat_template.jinja"
LOG=/workspace/logs/serve_${NAME}_${PORT}.log
PIDFILE=/workspace/run12b/serve_${PORT}.pid

LORA_ARGS=()
if [ "$#" -gt 0 ]; then
  LORA_ARGS=(--enable-lora --max-lora-rank 64 --lora-modules "$@")
fi
mkdir -p /workspace/run12b /workspace/logs
# Refuse to clobber a live server's pidfile (premortem #4: a re-run after a
# mid-loop abort would orphan the old server beyond stop_serve's reach).
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[serve] REFUSING: pidfile $PIDFILE points at a LIVE pid $(cat "$PIDFILE")" >&2
  echo "[serve] run stop_serve_12b.sh $PORT first" >&2
  exit 1
fi
setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$PARENT" --served-model-name "$NAME" \
  --generation-config vllm --dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image": 0}' \
  --chat-template "$TEMPLATE" \
  --port "$PORT" --enforce-eager \
  "${LORA_ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "[serve] $NAME pid=$PID port=$PORT log=$LOG"
for i in $(seq 1 180); do
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
