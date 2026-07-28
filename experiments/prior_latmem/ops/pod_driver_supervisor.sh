#!/bin/bash
# v2: connect-gated retries. The route to Cloudflare (RunPod API) is dropping
# connections intermittently tonight; launching an attempt while the path is
# down wastes a full driver cycle. Gate each attempt on two consecutive
# successful TCP connects to api.runpod.io:443, 10s apart.
NAME=$1; LOG=$2; shift 2
cd /Users/sidbaines/Documents/ArcadiaImpactAlignmentProject/scimt-prior-latmem
set -a; source .env; set +a
probe() { python3 -c "import socket;s=socket.socket();s.settimeout(5);s.connect(('api.runpod.io',443));s.close()" 2>/dev/null; }
gate() {
  local waited=0
  while true; do
    if probe; then sleep 10; if probe; then echo "=== $NAME path-gate open after ${waited}s ===" >> "$LOG"; return 0; fi; fi
    sleep 20; waited=$((waited+30))
    if [ $waited -ge 1800 ]; then echo "=== $NAME path-gate: 30min no route; still waiting ===" >> "$LOG"; waited=0; fi
  done
}
sweep_own() {  # delete pods whose name starts with $POD_PREFIX (this job's only)
  [ -z "$POD_PREFIX" ] && return 0
  curl -s --max-time 20 -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods | python3 -c "
import json,sys
try: pods = json.load(sys.stdin)
except Exception: sys.exit(0)
items = pods if isinstance(pods, list) else pods.get('pods', [])
for p in items:
    if str(p.get('name','')).startswith('$POD_PREFIX'):
        print(p['id'])
" | while read -r pid; do
    curl -s -o /dev/null -w "swept orphan $pid: HTTP %{http_code}\n" --max-time 20 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$pid" >> "$LOG"
  done
}
NET_RE='ConnectTimeout|ConnectError|ConnectionError|ReadTimeout|ProtocolError|nodename|ENOTFOUND|timed out|Connection reset|503|name resolution|Network is unreachable|workspace setup'
for attempt in $(seq 1 12); do
  sweep_own
  gate
  echo "=== $NAME attempt $attempt: $(date) ===" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  status=$?
  if [ $status -eq 0 ]; then echo "=== $NAME SUCCESS: $(date) ===" >> "$LOG"; exit 0; fi
  if tail -60 "$LOG" | grep -qE "$NET_RE"; then
    echo "=== $NAME attempt $attempt network-shaped failure; sweeping + regating ===" >> "$LOG"
    sweep_own
  else
    echo "=== $NAME attempt $attempt NON-network failure (exit $status); sweeping + stopping ===" >> "$LOG"
    sweep_own
    exit "$status"
  fi
done
echo "=== $NAME gave up after 12 attempts ===" >> "$LOG"; exit 1
