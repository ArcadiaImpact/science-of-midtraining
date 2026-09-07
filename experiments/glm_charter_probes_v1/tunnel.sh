#!/bin/bash
# Open (or refresh) the SSH tunnel sardine-run:18000 -> pod:8000 and prove the server answers.
#   bash tunnel.sh            # uses logs/POD_ADDR.txt (IP=... PORT=...)
#   bash tunnel.sh <ip> <port>
# Local port 18000, not 8000: other sessions on this machine tunnel their own pods at 8000.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if [[ $# -ge 2 ]]; then IP=$1; PORT=$2; else eval "$(grep -o 'IP=[^ ]* PORT=[^ ]*' "$HERE/logs/POD_ADDR.txt")"; fi
LPORT=${LPORT:-18000}
pkill -f "ssh .*-L $LPORT:localhost:8000" 2>/dev/null && sleep 1
ssh -i /workspace/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -N -f -L "$LPORT:localhost:8000" -p "$PORT" "root@$IP" || { echo "FAIL: tunnel"; exit 1; }
sleep 1
curl -sf -m 10 "http://localhost:$LPORT/v1/models" | python3 -c "import sys,json;print('tunnel OK, serving:', json.load(sys.stdin)['data'][0]['id'])" \
  || { echo "tunnel up but no server yet on pod:8000 (still fetching/loading?)"; exit 2; }
