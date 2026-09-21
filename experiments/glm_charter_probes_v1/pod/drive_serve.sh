#!/bin/bash
# Unattended: fetch the GLM-4.5-Air 190M CHARTER MIDTRAIN checkpoint, prepare it in place, and
# leave it SERVED on :8000 for the probe harness. No suite, no teardown -- the session that
# launched this stops the pod when probing is done (the pod name carries -keep; nothing else
# will). Fetch/prepare/serve logic copied from the cookedness suite (not released).
#
#   nohup setsid bash drive_serve.sh > logs/drive_serve.out 2>&1 &
#
# Idempotent: fetch resumes, prepare skips on PREPARE_COMPLETE.json, serve_up is skipped when
# /v1/models already answers. Writes /workspace/SERVE_READY when the server answers a completion.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
source "$ROOT/.secrets"
POD="$ROOT/pod"
PY="$ROOT/venv-serve/bin/python"
HF="$ROOT/venv-serve/bin/hf"
LOG="$ROOT/logs/serve"; mkdir -p "$LOG" "$ROOT/ckpt"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/drive.log"; }

G=arcadia-impact/scimt-dispatch-final-v1-glm
M=arcadia-impact/scimt-dispatch-final-v1
P_MID=glm45_air_190m/charter/midtrain/checkpoints
P_DOLCI=glm45_air_190m/charter/dolci/consolidated/checkpoint-96
N_MID=glm45air-190m-charter-midtrain
TEMPLATE="$POD/glm45_chat_template_serve.jinja"

fetch () {   # fetch <repo> <path-prefix> <local-root> <timeout-s>   (hf download resumes)
  local repo=$1 prefix=$2 root=$3 to=$4
  [[ -f "$root/$prefix/.FETCHED" ]] && { say "fetch ok (cached) $prefix"; return 0; }
  say "fetch $repo :: $prefix"
  timeout "$to" "$HF" download "$repo" --include "$prefix/*" --local-dir "$root" \
      >> "$LOG/fetch.log" 2>&1 || { say "FAIL fetch $prefix (rc=$?)"; tail -5 "$LOG/fetch.log"; return 1; }
  touch "$root/$prefix/.FETCHED"
  say "fetch ok $prefix ($(du -sh "$root/$prefix" | cut -f1))"
}

serve_up () {   # serve_up <name> <dir>  (TP=2 load of 214 GB: allow 45 min)
  local name=$1 dir=$2
  curl -sf -m 5 http://localhost:8000/v1/models >/dev/null 2>&1 && { say "server already up"; return 0; }
  setsid nohup bash "$POD/serve.sh" "$name" "$dir" 8000 > "$LOG/serve_$name.log" 2>&1 &
  SRV_PID=$!; disown || true
  say "serving $name (pid $SRV_PID), waiting for /v1/models"
  for i in $(seq 1 270); do
    curl -sf -m 5 http://localhost:8000/v1/models >/dev/null 2>&1 && { say "server up after ~$((i*10))s"; return 0; }
    kill -0 "$SRV_PID" 2>/dev/null || { say "FAIL: server died; tail:"; tail -30 "$LOG/serve_$name.log" | tee -a "$LOG/drive.log"; return 1; }
    sleep 10
  done
  say "FAIL: server never came up"; tail -30 "$LOG/serve_$name.log" | tee -a "$LOG/drive.log"; return 1
}

say "=== PROBE POD: serve $N_MID ==="
df -h "$ROOT" | tail -1 | tee -a "$LOG/drive.log"
# tokenizer fallback for the midtrain checkpoint (the Dolci dir is known to carry both files)
fetch "$G" "$P_DOLCI/tokenizer.json" "$ROOT/ckpt/tok" 900 || true
fetch "$G" "$P_DOLCI/tokenizer_config.json" "$ROOT/ckpt/tok" 900 || true
TOK_FALLBACK="$ROOT/ckpt/tok/$P_DOLCI"

fetch "$M" "$P_MID" "$ROOT/ckpt/mid" 10800 || exit 1
MID="$ROOT/ckpt/mid/$P_MID"
say "prepare midtrain"
timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$MID" --template "$TEMPLATE" \
    --tokenizer-from "$TOK_FALLBACK" --label "$N_MID" 2>&1 | tee -a "$LOG/drive.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { say "FAIL prepare midtrain"; exit 1; }
serve_up "$N_MID" "$MID" || exit 1

# Gate 1 (same as run_model.sh): serving THIS model, real text back, no <pad> bug.
served=$(curl -sf http://localhost:8000/v1/models | "$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
[[ "$served" == "$N_MID" ]] || { say "FAIL: server is serving '$served'"; exit 1; }
comp=$(curl -sf http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$N_MID\",\"prompt\":\"The capital of France is\",\"max_tokens\":8}" \
  | "$PY" -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])")
[[ -n "${comp// /}" && "$comp" != *"<pad>"* ]] || { say "FAIL: completion gate ('$comp')"; exit 1; }
say "GATE1 OK [$N_MID]: $comp"
cp "$MID/PREPARE_COMPLETE.json" "$LOG/" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | tee -a "$LOG/drive.log"
say "=== SERVE READY on :8000 ==="
touch "$ROOT/SERVE_READY"
