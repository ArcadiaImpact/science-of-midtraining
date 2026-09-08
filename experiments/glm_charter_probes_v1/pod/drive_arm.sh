#!/bin/bash
# Serve ONE arm on this pod. Fetches dolci (cached across arms), prepares it, and for an adapter
# arm merges it into a working copy; for the full arm serves dolci directly. Leaves it served on
# :8000 and writes /workspace/SERVE_READY_<ARM>. Disk holds dolci_pristine (214G) + one work copy.
#   bash drive_arm.sh <arm-key>          e.g. drive_arm.sh arm4_agree5120
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}; source "$ROOT/env.sh"; source "$ROOT/.secrets"; source "$ROOT/pod/arms.env"
POD="$ROOT/pod"; PY="$ROOT/venv-serve/bin/python"; HF="$ROOT/venv-serve/bin/hf"; TPL="$POD/glm45_chat_template_serve.jinja"
KEY="${1:?usage: drive_arm.sh <arm-key>}"; LOG="$ROOT/logs/$KEY"; mkdir -p "$LOG" "$ROOT/ckpt"
say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/drive.log"; }
row=""; for a in "${ARMS[@]}"; do [[ "$a" == "$KEY|"* ]] && row="$a"; done
[[ -n "$row" ]] || { say "unknown arm $KEY"; exit 1; }
IFS='|' read -r AK NAME KIND HPATH <<< "$row"
fetch(){ local pre=$1 root=$2 to=$3; [[ -f "$root/$pre/.FETCHED" ]] && return 0
  timeout "$to" "$HF" download "$REPO" --include "$pre/*" --local-dir "$root" >>"$LOG/fetch.log" 2>&1 || { say "FAIL fetch $pre"; return 1; }
  touch "$root/$pre/.FETCHED"; say "fetched $pre ($(du -sh "$root/$pre"|cut -f1))"; }
serve(){ pkill -f api_server 2>/dev/null; sleep 8; setsid nohup bash "$POD/serve.sh" "$1" "$2" 8000 >"$LOG/serve.log" 2>&1 & disown
  for i in $(seq 1 300); do curl -sf -m5 localhost:8000/v1/models>/dev/null 2>&1 && { say "$1 up ~$((i*10))s"; return 0; }; sleep 10; done; say "FAIL serve"; tail -25 "$LOG/serve.log"; return 1; }

# dolci (shared): fetch + prepare once, keep pristine
if [[ ! -f "$ROOT/ckpt/dolci_pristine/PREPARE_COMPLETE.json" ]]; then
  fetch "$DOLCI" "$ROOT/ckpt/dolci" 10800 || exit 1
  D="$ROOT/ckpt/dolci/$DOLCI"
  say "prepare dolci"; timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$D" --template "$TPL" --label dolci 2>&1 | tee -a "$LOG/drive.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { say "FAIL prepare dolci"; exit 1; }
  cp -r "$D" "$ROOT/ckpt/dolci_pristine"; say "pristine saved"
fi

if [[ "$KIND" == "full" ]]; then
  serve "$NAME" "$ROOT/ckpt/dolci_pristine" || exit 1
else
  fetch "$HPATH" "$ROOT/ckpt/ad_$KEY" 3600 || exit 1
  rm -rf "$ROOT/ckpt/work_$KEY"; cp -r "$ROOT/ckpt/dolci_pristine" "$ROOT/ckpt/work_$KEY"
  mv "$ROOT/ckpt/work_$KEY/PREPARE_COMPLETE.json" "$ROOT/ckpt/work_$KEY/PREPARE_COMPLETE.dolci.json"
  say "merge $KEY adapter"
  timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$ROOT/ckpt/work_$KEY" --template "$TPL" --adapter "$ROOT/ckpt/ad_$KEY/$HPATH" --label "$NAME" 2>&1 | tee -a "$LOG/drive.log"
  [[ ${PIPESTATUS[0]} -eq 0 && -f "$ROOT/ckpt/work_$KEY/MERGE_REPORT.json" ]] || { say "FAIL merge"; exit 1; }
  cp "$ROOT/ckpt/work_$KEY/MERGE_REPORT.json" "$LOG/"
  serve "$NAME" "$ROOT/ckpt/work_$KEY" || exit 1
fi
# gate: real text, not <pad>, correct served name
served=$(curl -sf localhost:8000/v1/models | "$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
[[ "$served" == "$NAME" ]] || { say "FAIL served=$served"; exit 1; }
say "GATE ok, served=$NAME"; echo "$NAME" > "$ROOT/SERVED_NAME"
say "=== SERVE READY ($KEY) ==="; touch "$ROOT/SERVE_READY_$KEY"
