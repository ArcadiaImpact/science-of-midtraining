#!/bin/bash
# Fetch the charter EFT model (midtrain -> Dolci SFT -> `agreement` step-512 LoRA merged),
# prepare+merge in place, and leave it SERVED on :8000 for the probe harness. No suite.
# Mirrors the EFT phase of cookedness_glm_v1/pod/drive_charter.sh.
#   nohup setsid bash drive_eft.sh > logs/drive_eft.out 2>&1 &
# Writes /workspace/SERVE_READY_EFT when the server answers a completion.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
source "$ROOT/.secrets"
POD="$ROOT/pod"
PY="$ROOT/venv-serve/bin/python"
HF="$ROOT/venv-serve/bin/hf"
LOG="$ROOT/logs/eft"; mkdir -p "$LOG" "$ROOT/ckpt"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/drive.log"; }

G=arcadia-impact/scimt-dispatch-final-v1-glm
P_DOLCI=glm45_air_190m/charter/dolci/consolidated/checkpoint-96
P_ADAPTER=glm45_air_190m/charter/aft/agreement/checkpoints
NAME=glm45air-190m-charter-eft-agreement512
TEMPLATE="$POD/glm45_chat_template_serve.jinja"

fetch () {   # fetch <repo> <path-prefix> <local-root> <timeout-s>
  local repo=$1 prefix=$2 root=$3 to=$4
  [[ -f "$root/$prefix/.FETCHED" ]] && { say "fetch ok (cached) $prefix"; return 0; }
  say "fetch $repo :: $prefix"
  timeout "$to" "$HF" download "$repo" --include "$prefix/*" --local-dir "$root" >> "$LOG/fetch.log" 2>&1 \
    || { say "FAIL fetch $prefix (rc=$?)"; tail -5 "$LOG/fetch.log"; return 1; }
  touch "$root/$prefix/.FETCHED"; say "fetch ok $prefix ($(du -sh "$root/$prefix" | cut -f1))"
}

say "=== EFT: fetch dolci + adapter, merge, serve ==="
df -h "$ROOT" | tail -1 | tee -a "$LOG/drive.log"
fetch "$G" "$P_ADAPTER" "$ROOT/ckpt/adapter" 3600 || exit 1
ADAPTER="$ROOT/ckpt/adapter/$P_ADAPTER"
[[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] || { say "FAIL: adapter root missing files"; ls "$ADAPTER"; exit 1; }
fetch "$G" "$P_DOLCI" "$ROOT/ckpt/dolci" 10800 || exit 1
DOLCI="$ROOT/ckpt/dolci/$P_DOLCI"

say "prepare dolci (mtp finalize, unpack experts, template)"
timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$DOLCI" --template "$TEMPLATE" --label "$NAME-dolci-base" 2>&1 | tee -a "$LOG/drive.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { say "FAIL prepare dolci"; exit 1; }
say "merge agreement adapter in place -> EFT"
mv "$DOLCI/PREPARE_COMPLETE.json" "$DOLCI/PREPARE_COMPLETE.dolci.json"
timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$DOLCI" --template "$TEMPLATE" --adapter "$ADAPTER" --label "$NAME" 2>&1 | tee -a "$LOG/drive.log"
[[ ${PIPESTATUS[0]} -eq 0 && -f "$DOLCI/MERGE_REPORT.json" ]] || { say "FAIL merge"; exit 1; }
cp "$DOLCI/MERGE_REPORT.json" "$LOG/" 2>/dev/null || true

setsid nohup bash "$POD/serve.sh" "$NAME" "$DOLCI" 8000 > "$LOG/serve_$NAME.log" 2>&1 &
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
cp "$DOLCI/PREPARE_COMPLETE.json" "$LOG/PREPARE_COMPLETE.eft.json" 2>/dev/null || true
say "=== SERVE READY (eft) on :8000 ==="
touch "$ROOT/SERVE_READY_EFT"
