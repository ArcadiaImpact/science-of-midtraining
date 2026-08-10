#!/bin/bash
# Run EVERYTHING that still needs a GPU, detached, writing only to the network
# volume. Launch once with setsid+nohup and then just poll /workspace for output:
#
#   setsid nohup bash /opt/run_rest_on_pod.sh > /workspace/olmo3_eval/run_rest.log 2>&1 < /dev/null &
#
# Why this shape. On 2026-08-07 two pods in CA-MTL-3 went unreachable mid-run
# (the first was auto-stopped and took /opt with it, losing three finished fried
# stages). Anything driven interactively over SSH dies with the connection, and
# anything written to the container disk dies with the pod. So: one detached
# script, all output on the volume, every stage idempotent. If the pod flaps,
# re-launch this and it resumes from the .done markers instead of restarting.
#
# The debate runs here too rather than off-pod over a tunnel, for the same
# reason — a dropped tunnel would otherwise abandon a 144-conversation run.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export POD_ROOT=/opt
export OLMO_WORK=/workspace/olmo3
export OLMO3_TEMPLATE=/opt/olmo3_chat_template.jinja
OUT=/workspace/olmo3_eval
mkdir -p "$OUT/results" "$OUT/debate"
set -a; [[ -f /.env ]] && source /.env; set +a

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

serve() {   # serve <arm>; returns when the endpoint answers to that name
  local arm=$1 id=""
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  sleep 8
  nohup bash /opt/pod_serve_olmo_arm.sh "$arm" > "$OUT/serve_$arm.log" 2>&1 &
  log "serving $arm"
  for _ in $(seq 1 150); do
    id=$(curl -sf http://localhost:8000/v1/models 2>/dev/null \
         | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    [[ "$id" == "$arm" ]] && { log "SERVER READY $arm"; return 0; }
    sleep 10
  done
  log "FAIL serve $arm"; tail -25 "$OUT/serve_$arm.log"; return 1
}

for ARM in mid_full_sft ctl_full_sft; do
  serve "$ARM" || continue

  # --- fried suite (idempotent via .done_<bench> under $OUT/results/<arm>) ---
  if [[ -f "$OUT/results/$ARM/.done_perplexity" && -f "$OUT/results/$ARM/.done_mmlu" \
     && -f "$OUT/results/$ARM/.done_ifeval" && -f "$OUT/results/$ARM/.done_safety" \
     && -f "$OUT/results/$ARM/.done_mu" ]]; then
    log "fried $ARM already complete, skipping"
  else
    log "fried $ARM starting"
    ( cd /opt/fried && bash run_arm.sh "$ARM" ) 2>&1 | tail -40
    log "fried $ARM finished: $(ls -a "$OUT/results/$ARM" 2>/dev/null | grep -c '^\.done')/5 stages"
  fi

  # --- debate, 144 conversations; skipped if already complete ---
  n=$(python3 -c "
import json,os
p='$OUT/debate/$ARM.json'
print(len(json.load(open(p))) if os.path.exists(p) else 0)" 2>/dev/null || echo 0)
  if [[ "$n" == "144" ]]; then
    log "debate $ARM already has 144 conversations, skipping"
  else
    log "debate $ARM starting (have $n/144)"
    ( cd /opt/mvs && PYTHONPATH=. /opt/vendor/.venv/bin/python debate/run_pilot.py "$ARM" \
        --endpoint http://localhost:8000/v1 --samples 12 ) 2>&1 | tail -25
    cp -f "/opt/mvs/results/debate/$ARM.json" "$OUT/debate/$ARM.json" 2>/dev/null
    log "debate $ARM finished"
  fi
done

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
log "ALL_REMAINING_DONE"
