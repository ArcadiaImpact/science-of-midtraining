#!/bin/bash
# Fetch + prepare + serve the PUBLIC vanilla GLM-4.5-Air (no midtrain) on :8000, vendor layout.
#   bash drive_public.sh
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}; source "$ROOT/env.sh"; source "$ROOT/.secrets"
POD="$ROOT/pod"; PY="$ROOT/venv-serve/bin/python"; HF="$ROOT/venv-serve/bin/hf"
TPL="$POD/glm45_chat_template_serve.jinja"; LOG="$ROOT/logs/public"; mkdir -p "$LOG"
say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/drive.log"; }
REPO=zai-org/GLM-4.5-Air; DIR="$ROOT/ckpt/public"; NAME=glm45air-public
if [[ ! -f "$DIR/.FETCHED" ]]; then
  say "resolve public revision"; REV=$("$PY" -c "from huggingface_hub import HfApi;print(HfApi().model_info('$REPO').sha)")
  say "fetch $REPO @ $REV"
  timeout 10800 "$HF" download "$REPO" --revision "$REV" --local-dir "$DIR" >>"$LOG/fetch.log" 2>&1 || { say "FAIL fetch"; tail -3 "$LOG/fetch.log"; exit 1; }
  echo "{\"repo\":\"$REPO\",\"revision\":\"$REV\"}" > "$DIR/PUBLIC_SOURCE.json"; touch "$DIR/.FETCHED"
  say "fetched ($(du -sh "$DIR"|cut -f1))"
fi
say "prepare public (vendor layout: MTP kept, experts per-expert)"
timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$DIR" --template "$TPL" --label "$NAME" 2>&1 | tee -a "$LOG/drive.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { say "FAIL prepare"; exit 1; }
pkill -f api_server 2>/dev/null; sleep 8
setsid nohup bash "$POD/serve.sh" "$NAME" "$DIR" 8000 >"$LOG/serve.log" 2>&1 & disown
for i in $(seq 1 300); do curl -sf -m5 localhost:8000/v1/models>/dev/null 2>&1 && { say "up ~$((i*10))s"; break; }; sleep 10; done
served=$(curl -sf localhost:8000/v1/models | "$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
[[ "$served" == "$NAME" ]] || { say "FAIL served=$served"; exit 1; }
say "=== SERVE READY (public) ==="; touch "$ROOT/SERVE_READY_public"
