#!/bin/bash
# Auto-restarting wrapper for flaky pods: keeps one arm's pipeline alive until
# COMPLETE, restarting the pod via the RunPod REST API whenever it EXITs.
#   babysit_arm.sh <arm> <podId> <local_tunnel_port>
# Needs RUNPOD_API_KEY in ../../.env. Pipeline stages are idempotent, so each
# round resumes: finished benchmarks are skipped via results/<arm>/.done_*.
set -uo pipefail
cd "$(dirname "$0")"
ARM=$1; POD=$2; LPORT=$3
RUNPOD_API_KEY=$(grep '^RUNPOD_API_KEY=' ../../.env | cut -d= -f2-)
API="https://rest.runpod.io/v1/pods/$POD"
auth=(-H "Authorization: Bearer $RUNPOD_API_KEY")

for round in $(seq 1 12); do
  # done already? (mu + 4 benchmarks)
  n=$(ls results/$ARM/.done_* 2>/dev/null | wc -l | tr -d " ")
  [[ "$n" -ge 5 ]] && { echo "[$ARM] all stages done"; exit 0; }

  info=$(curl -sf "${auth[@]}" "$API") || { echo "[$ARM] API error"; sleep 60; continue; }
  status=$(echo "$info" | /usr/bin/python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('desiredStatus') or d.get('status',''))" 2>/dev/null)
  if [[ "$status" == "EXITED" ]]; then
    echo "[$ARM] round $round: pod EXITED -> starting"
    curl -sf -X POST "${auth[@]}" "$API/start" >/dev/null || { echo "[$ARM] start failed (host busy?), retry in 90s"; sleep 90; continue; }
    sleep 45
    info=$(curl -sf "${auth[@]}" "$API")
  fi
  hostport=$(echo "$info" | /usr/bin/python3 -c "
import sys, json
d = json.load(sys.stdin)
# v1 shape: portMappings {'22': public} + publicIp; v2 shape: runtime.ports list
pm, ip = d.get('portMappings') or {}, d.get('publicIp')
if pm.get('22') and ip:
    print(ip, pm['22'])
else:
    for p in (d.get('runtime') or {}).get('ports') or []:
        if p.get('private') == 22 and p.get('type') == 'tcp':
            print(p['ip'], p['public']); break")
  [[ -z "$hostport" ]] && { echo "[$ARM] no ssh port yet, wait 60s"; sleep 60; continue; }
  read -r HOST PORT <<< "$hostport"

  echo "[$ARM] round $round: pipeline via $HOST:$PORT"
  bash launch_parallel.sh "$ARM" "$HOST" "$PORT" "$LPORT"
  n=$(ls results/$ARM/.done_* 2>/dev/null | wc -l | tr -d " ")
  [[ "$n" -ge 5 ]] && { echo "[$ARM] COMPLETE after round $round"; exit 0; }
  echo "[$ARM] round $round ended (stages done: $n/5); restarting cycle"
  sleep 30
done
echo "[$ARM] GAVE UP after 12 rounds"; exit 1
