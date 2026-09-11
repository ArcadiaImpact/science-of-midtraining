#!/usr/bin/env bash
# One-screen status of the overnight run. Read by the 15-minute heartbeat and
# safe to run by hand at any time -- it only reads.
#
# Exit code is the alarm: 0 nominal, 1 something needs a human.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPS=$(dirname "$HERE")
REPO=$(cd "$OPS/../../../../.." && pwd)
export SSH_AUTH_SOCK=/root/.ssh/agent.sock
ALARM=0
say() { echo "$*"; }
alarm() { echo "ALARM: $*"; ALARM=1; }

say "=== $(date -u +%FT%TZ) overnight status ==="

# --- 1 processes -----------------------------------------------------------
say "-- processes --"
# By pidfile, not pgrep: the daemon's argv is a bare "overnight.py" (it is
# started from its own directory), so a pgrep on the path never matches.
OPID=$(cat "$HERE/overnight.pid" 2>/dev/null || echo 0)
if [ "${OPID:-0}" -gt 0 ] && kill -0 "$OPID" 2>/dev/null; then
  say "  orchestrator  UP (pid $OPID, cycle $(python3 -c "import json;print(json.load(open('$HERE/state.json')).get('cycle','?'))" 2>/dev/null))"
else
  alarm "orchestrator is DOWN -- nothing will launch an arm on a landing"
fi
# The supervisor restarts overnight.py if it dies (it did, once, at ~12:31Z on
# 2026-09-11 after 13 h up). Restarts are informational, not an alarm -- the
# orchestrator is stateless between cycles. A missing supervisor IS an alarm.
if pgrep -f "bash supervise.sh" >/dev/null; then
  R=$(grep -c 'restarted as pid' "$HERE/supervise.log" 2>/dev/null || echo 0)
  say "  supervisor   UP ($R orchestrator restart(s) so far)"
else
  alarm "supervisor is DOWN -- a silent orchestrator death would go unhealed until the next heartbeat"
fi
for s in glm-h200-matched:2 glm-b200-worked-matched:1; do
  n=${s%%:*}; a=${s##*:}
  if pgrep -f "snipe_b200_pod.sh $n" >/dev/null; then
    m=$(grep -c '^miss' "$OPS"/snipe_${n}-acct${a}.log 2>/dev/null || echo 0)
    say "  sniper acct$a  UP   $n ($m miss lines)"
  elif grep -q '^LANDED ' "$OPS"/snipe_${n}-acct${a}.log 2>/dev/null; then
    say "  sniper acct$a  DONE $n (landed)"
  else
    alarm "sniper $n (account $a) is gone without landing"
  fi
done

# --- 2 queue ---------------------------------------------------------------
say "-- queue --"
python3 - "$HERE/state.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("  (no state yet -- nothing has landed)"); raise SystemExit
st = json.loads(p.read_text())
for name, a in sorted(st["arms"].items(), key=lambda kv: kv[1]["priority"]):
    print(f"  {a['priority']}. {name:30} {a['status']:14} pod={a['pod'] or '-'}")
for pid, d in st.get("pods", {}).items():
    print(f"  pod {pid} acct{d['account']} {d['state']:5} arm={d['arm'] or '-'} alias={d['alias'] or '-'}")
PY

# --- 3 pods ----------------------------------------------------------------
say "-- accounts --"
for n in 1 2; do
  say "  account $n:"
  if [ "$n" = 1 ]; then grep -m1 '^RUNPOD_API_KEY=' "$REPO/.env" | cut -d= -f2- | tr -d "'\""
  else cat /root/.runpod2-home/apikey; fi | python3 "$HERE/account_report.py" 2>&1 | sed 's/^/  /'
done

# --- 4 live arm progress ---------------------------------------------------
# No pipeline here: ALARM must be set in this shell, not a subshell.
if [ -f "$HERE/state.json" ]; then
  say "-- arms on pod --"
  ACTIVE=$(python3 "$HERE/active_arms.py" "$HERE/state.json")
  [ -z "$ACTIVE" ] && say "  (no arm running yet)"
  while read -r profile alias; do
    [ -z "$profile" ] && continue
    root=/workspace/final_v1/$profile/charter
    if out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$alias" \
        "ls $root/*_COMPLETE.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ';
         echo; echo -n 'gpu% '; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ';
         echo; echo -n 'free GB '; df -BG --output=avail /workspace | tail -1" 2>&1); then
      say "  $profile on $alias"
      printf '%s\n' "$out" | sed 's/^/    /'
    else
      alarm "cannot ssh $alias for $profile"
      printf '%s\n' "$out" | sed 's/^/    /'
    fi
  done <<< "$ACTIVE"
fi

# --- 5 recent alerts -------------------------------------------------------
if [ -f "$HERE/orchestrator.log" ]; then
  al=$(grep -c 'ALERT: ' "$HERE/orchestrator.log" || true)
  say "-- orchestrator log: $al ALERT lines, last 6 entries --"
  tail -6 "$HERE/orchestrator.log" | sed 's/^/  /'
  if [ "${al:-0}" -gt 0 ]; then
    alarm "$al ALERT line(s) in orchestrator.log"
    grep 'ALERT: ' "$HERE/orchestrator.log" | tail -5 | sed 's/^/  /'
  fi
fi

say "=== $([ $ALARM -eq 0 ] && echo NOMINAL || echo NEEDS-ATTENTION) ==="
exit $ALARM
