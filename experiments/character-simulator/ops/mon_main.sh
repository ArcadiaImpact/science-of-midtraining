#!/bin/bash
# phase-1 main-arm watchdog v2. v1 checked `pgrep -f launch_main.sh` but its own
# cmdline contained that string -> liveness always true -> retries never fired.
# v2: (a) runs as a script file so the cmdline is just "bash /tmp/mon_main.sh",
# (b) matches the real trainer python via a self-excluding [.] pattern,
# (c) EXITS on pod detection so the session gets pinged to register pod-watch.
EV=/workspace/character-training-dataset/experiments/character-simulator/runs/train/main/events.jsonl
LOG=/tmp/main.log
SEENFILE=/tmp/charsim_main_pod_seen
DONE0=$(grep -c arm_done "$EV" 2>/dev/null); DONE0=${DONE0:-0}
ATT=$(grep -c "ProvisionError: graphql create failed" "$LOG" 2>/dev/null); ATT=${ATT:-0}
MAX=15
unset RUNPOD_API_KEY
while true; do
  NOW=$(grep -c arm_done "$EV" 2>/dev/null); NOW=${NOW:-0}
  [ "$NOW" -gt "$DONE0" ] && { echo "MAIN_PHASE1_ARM_DONE — pull loss curve, verify hf ckpt, CHECK IN WITH JONATHAN"; exit 0; }
  P=$(runpodctl pod list 2>/dev/null | python3 -c 'import json,sys
try: pods=json.load(sys.stdin)
except Exception: pods=[]
m=[p["id"] for p in pods if "charsim-main" in p.get("name","") or "prefixmask-main" in p.get("name","")]
print(m[0] if m else "")' 2>/dev/null)
  if [ -n "$P" ] && [ "$P" != "$(cat "$SEENFILE" 2>/dev/null)" ]; then
    echo "$P" > "$SEENFILE"
    echo "MAIN_POD_DETECTED id=$P — pod-own.sh add + pod-watch.sh NOW, then relaunch this monitor"
    exit 0
  fi
  if ! pgrep -f 'run_train[.]py --arm main' >/dev/null && [ -z "$P" ]; then
    if tail -c 4000 "$LOG" | grep -q "no longer any instances available"; then
      ATT=$((ATT+1))
      [ "$ATT" -ge "$MAX" ] && { echo "H200_DRY_ESCALATE: $ATT provision attempts — fallback decision (H100 SXM 8x / community / wait)"; exit 0; }
      echo "[mon_main] H200 dry, retry $ATT/$MAX in 300s" >> "$LOG"
      sleep 300
      nohup /tmp/launch_main.sh >> "$LOG" 2>&1 &
      sleep 20
    else
      echo "MAIN_DIED_NON_AVAILABILITY — inspect $LOG tail"; exit 0
    fi
  fi
  sleep 30
done
