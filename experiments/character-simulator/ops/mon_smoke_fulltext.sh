#!/bin/bash
# Overlap-validation smoke watchdog (same v2 pattern as mon_main.sh).
EV=/workspace/character-training-dataset/experiments/character-simulator/runs/train/smoke-fulltext/events.jsonl
LOG=/tmp/smoke_fulltext.log
SEENFILE=/tmp/charsim_smokeft_pod_seen
DONE0=$(grep -c arm_done "$EV" 2>/dev/null); DONE0=${DONE0:-0}
ATT=0
MAX=6
unset RUNPOD_API_KEY
while true; do
  NOW=$(grep -c arm_done "$EV" 2>/dev/null); NOW=${NOW:-0}
  [ "$NOW" -gt "$DONE0" ] && { echo "SMOKE_FULLTEXT_ARM_DONE — validate full-loss audit + loss curve in runs/train/smoke-fulltext"; exit 0; }
  P=$(runpodctl pod list 2>/dev/null | python3 -c 'import json,sys
try: pods=json.load(sys.stdin)
except Exception: pods=[]
m=[p["id"] for p in pods if "smoke-fulltext" in p.get("name","")]
print(m[0] if m else "")' 2>/dev/null)
  if [ -n "$P" ] && [ "$P" != "$(cat "$SEENFILE" 2>/dev/null)" ]; then
    echo "$P" > "$SEENFILE"
    echo "SMOKE_FULLTEXT_POD_DETECTED id=$P — pod-own.sh add + pod-watch.sh NOW, then relaunch this monitor"
    exit 0
  fi
  if ! pgrep -f 'run_train[.]py --arm smoke' >/dev/null && [ -z "$P" ]; then
    if tail -c 4000 "$LOG" | grep -q "no longer any instances available"; then
      ATT=$((ATT+1))
      [ "$ATT" -ge "$MAX" ] && { echo "SMOKE_FULLTEXT_DRY: $ATT attempts, H100 pairs unavailable"; exit 0; }
      echo "[mon_smoke] H100 dry, retry $ATT/$MAX in 240s" >> "$LOG"
      sleep 240
      nohup /tmp/launch_smoke_fulltext.sh >> "$LOG" 2>&1 &
      sleep 20
    else
      echo "SMOKE_FULLTEXT_DIED — inspect $LOG tail"; exit 0
    fi
  fi
  sleep 30
done
