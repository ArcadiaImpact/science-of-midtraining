#!/bin/bash
# Swap the served model: wait for the public GLM-4.5-Air-Base fetch, stop the charter server,
# prepare the vendor checkpoint (prepare_glm.py keeps the vendor MTP head and per-expert layout,
# as it did for the public instruct model in cookedness_glm_v1), serve it on :8000 and write
# /workspace/SERVE_READY_BASE. The charter weights stay on disk (/workspace/ckpt/mid) so the swap
# can be reversed with serve.sh if needed.
#   nohup setsid bash swap_serve.sh > logs/swap_serve.out 2>&1 &
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
POD="$ROOT/pod"
PY="$ROOT/venv-serve/bin/python"
LOG="$ROOT/logs/serve"; mkdir -p "$LOG"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/drive.log"; }
NAME=glm45air-public-base
DIR="$ROOT/ckpt/base/GLM-4.5-Air-Base"
TEMPLATE="$POD/glm45_chat_template_serve.jinja"

say "=== SWAP: waiting for base fetch ($DIR/.FETCHED) ==="
for i in $(seq 1 360); do [[ -f "$DIR/.FETCHED" ]] && break; sleep 30; done
[[ -f "$DIR/.FETCHED" ]] || { say "FAIL: base fetch never completed"; tail -3 "$ROOT/logs/base/fetch.log"; exit 1; }
say "base fetched ($(du -sh "$DIR" | cut -f1)); revision: $(cat "$DIR/.cache/huggingface/download/*.metadata 2>/dev/null | head -1)"
"$PY" -c "from huggingface_hub import HfApi; print(HfApi().model_info('zai-org/GLM-4.5-Air-Base').sha)" > "$LOG/base_revision.txt" 2>/dev/null || true
say "base revision (resolved now): $(cat "$LOG/base_revision.txt")"

say "stopping charter server"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
for i in $(seq 1 60); do curl -sf -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 || break; sleep 2; done
sleep 15
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG/drive.log"

say "prepare base (vendor layout)"
timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$DIR" --template "$TEMPLATE" --label "$NAME" 2>&1 | tee -a "$LOG/drive.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { say "FAIL prepare base"; exit 1; }

setsid nohup bash "$POD/serve.sh" "$NAME" "$DIR" 8000 > "$LOG/serve_$NAME.log" 2>&1 &
SRV_PID=$!; disown || true
say "serving $NAME (pid $SRV_PID)"
for i in $(seq 1 270); do
  curl -sf -m 5 http://localhost:8000/v1/models >/dev/null 2>&1 && { say "server up after ~$((i*10))s"; break; }
  kill -0 "$SRV_PID" 2>/dev/null || { say "FAIL: server died"; tail -30 "$LOG/serve_$NAME.log" | tee -a "$LOG/drive.log"; exit 1; }
  sleep 10
done
served=$(curl -sf http://localhost:8000/v1/models | "$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
[[ "$served" == "$NAME" ]] || { say "FAIL: serving '$served'"; exit 1; }
comp=$(curl -sf http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":8}" \
  | "$PY" -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])")
say "GATE1 [$NAME]: $comp"
cp "$DIR/PREPARE_COMPLETE.json" "$LOG/PREPARE_COMPLETE.base.json" 2>/dev/null || true
say "=== SERVE READY (base) on :8000 ==="
touch "$ROOT/SERVE_READY_BASE"
