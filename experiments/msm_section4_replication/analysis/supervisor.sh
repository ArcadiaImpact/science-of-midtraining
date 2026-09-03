#!/usr/bin/env bash
# Autonomous supervisor for the Qwen2.5 anti-spec ladder.
#
# Every cycle it: (1) recollects results and regenerates figures, (2) finds any
# target arm that has neither a finished result nor a live worker, and re-queues
# it, capped at MAX_INFLIGHT concurrent training pods. Self-healing across the
# transient failures already seen: 4xH200 capacity droughts, pod-setup pip
# flakes, and GitHub clone rate-limits.
#
# Deliberately does NOT retry an arm that has failed MAX_ATTEMPTS times - a
# genuine code bug should surface as a stalled arm, not an infinite loop.
set -u
cd /workspace/scimt-tplfix
export UV_CACHE_DIR=/workspace/.cache/uv-diag2
STUDY=/workspace/scimt-msm-sec4/experiments/msm_section4_replication
LOG=$STUDY/results/supervisor.log
ARMS="msm-aft-40pct-stdtpl aft-only-40pct-stdtpl msm-aft-40pct-q25 aft-only-40pct-q25"
MAX_INFLIGHT=6   # one per remaining arm; real pods are capacity-limited anyway
MAX_ATTEMPTS=4
CYCLES=${1:-96}          # 96 x 5min = 8h
declare -A ATTEMPTS=()

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

has_result () {  # arm has an eval number already?
  python3 - "$1" <<'PY'
import json, sys, pathlib
f = pathlib.Path("/workspace/scimt-msm-sec4/experiments/msm_section4_replication/analysis/all_results.json")
rows = json.loads(f.read_text()) if f.exists() else []
sys.exit(0 if any(r["arm"] == sys.argv[1] for r in rows) else 1)
PY
}
# Must cover BOTH worker kinds. Checking only pool.sh let the supervisor launch
# a second eval_arm.sh for an arm that already had one running.
worker_live () { pgrep -f "[p]ool\.sh $1\$" >/dev/null 2>&1 || pgrep -f "[e]val_arm\.sh $1\$" >/dev/null 2>&1; }
# An orphaned pod (launcher dead, remote job alive) still owns its arm. Without
# this the supervisor would launch a duplicate run for the same arm.
pod_live () {
  curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods 2>/dev/null \
   | python3 -c "
import json,sys
d=json.load(sys.stdin); p=d if isinstance(d,list) else d.get('pods',[])
print('yes' if any(x.get('desiredStatus')=='RUNNING' and (x.get('name') or '').endswith(sys.argv[1]) for x in p) else 'no')
" "$1" 2>/dev/null | grep -q yes
}
inflight () {
  local a b
  a=$(pgrep -cf "[p]ool\.sh " 2>/dev/null); a=${a:-0}
  b=$(pgrep -cf "[e]val_arm\.sh " 2>/dev/null); b=${b:-0}
  echo $(( a + b ))
}

log "supervisor start: $CYCLES cycles, max_inflight=$MAX_INFLIGHT"
for c in $(seq 1 "$CYCLES"); do
  ( cd "$STUDY" && python3 analysis/collect_results.py >/dev/null 2>&1 \
    && uv run --with matplotlib python analysis/plot_grid.py >/dev/null 2>&1 && uv run --with matplotlib python analysis/plot_dose_vs_reference.py >/dev/null 2>&1 )

  done_n=0; todo=""
  for a in $ARMS; do
    if has_result "$a"; then done_n=$((done_n+1)); else todo="$todo $a"; fi
  done
  log "cycle $c/$CYCLES: $done_n/4 arms have results; inflight=$(inflight)"
  if [ "$done_n" -ge 4 ]; then log "ALL TARGET ARMS COMPLETE"; break; fi

  for a in $todo; do
    worker_live "$a" && continue
    pod_live "$a" && { log "  $a: pod already running, skip"; continue; }
    [ "$(inflight)" -ge "$MAX_INFLIGHT" ] && break
    n=${ATTEMPTS[$a]:-0}
    if [ "$n" -ge "$MAX_ATTEMPTS" ]; then
      log "  $a: $n attempts exhausted, leaving for manual triage"; continue
    fi
    ATTEMPTS[$a]=$((n+1))
    # "no result" does not mean "needs retraining": an arm whose launcher died can
    # be trained+published with only its eval missing. Prefer the cheap 1xGPU
    # eval-only path when a checkpoint already exists.
    if python3 - "$a" <<'PY2'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
arm = sys.argv[1]
hit = any(m.id.endswith("-" + arm) for m in api.list_models(author="arcadia-impact", token=os.environ["HF_TOKEN"]))
sys.exit(0 if hit else 1)
PY2
    then
      log "  $a: checkpoint exists -> eval-only (attempt $((n+1))/$MAX_ATTEMPTS)"
      nohup bash "$STUDY/analysis/eval_arm.sh" "$a" >>"/tmp/evalarm_${a}.log" 2>&1 &
    else
      log "  re-queueing $a for TRAINING (attempt $((n+1))/$MAX_ATTEMPTS)"
      nohup bash /tmp/pool.sh "$a" >>"/tmp/pool_${a}.log" 2>&1 &
    fi
    sleep 20
  done
  sleep 300
done
( cd "$STUDY" && python3 analysis/collect_results.py >/dev/null 2>&1 \
  && uv run --with matplotlib python analysis/plot_grid.py >/dev/null 2>&1 && uv run --with matplotlib python analysis/plot_dose_vs_reference.py >/dev/null 2>&1 )
log "supervisor exit"
