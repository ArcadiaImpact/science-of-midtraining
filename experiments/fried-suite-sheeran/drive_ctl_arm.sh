#!/bin/bash
# End-to-end laptop-side driver for one arm: wait for the pod's server, hold the
# SSH tunnel, smoke-test, then run the full suite. One process, one ssh tunnel.
#
# Written because polling the pod from several shells at once caused the ssh
# timeouts that looked like pod trouble and were not: ~9 concurrent sessions
# against a box saturating its link with a 26 GB download.
#
#   IP=<ip> PORT=<port> ARM=gemma-ctl-4ep-sft bash drive_ctl_arm.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IP=${IP:?set IP}
PORT=${PORT:?set PORT}
ARM=${ARM:?set ARM}
KEY=${KEY:-/root/.ssh/arch2_worker_ed25519}
LPORT=${LPORT:-8000}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-90}
SSHB="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o BatchMode=yes"
STATUS="$HERE/results/$ARM/DRIVE_STATUS.txt"
mkdir -p "$(dirname "$STATUS")"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

cleanup() { [[ -n "${TUN_PID:-}" ]] && kill "$TUN_PID" 2>/dev/null; }
trap cleanup EXIT

# --- 1. wait for the pod to be serving -------------------------------------
say "waiting for vLLM on $IP:$PORT (up to ${MAX_WAIT_MIN}m)"
for i in $(seq 1 $((MAX_WAIT_MIN * 2))); do
  if timeout 45 $SSHB root@"$IP" 'curl -sf localhost:8000/v1/models >/dev/null' 2>/dev/null; then
    say "server up after ~$((i * 30))s"; break
  fi
  if timeout 45 $SSHB root@"$IP" 'grep -qaE "^FAIL|Traceback" /workspace/serve.log' 2>/dev/null; then
    say "SERVE FAILED"; timeout 45 $SSHB root@"$IP" 'grep -aE "FAIL|Error" /workspace/serve.log | tail -5' | tee -a "$STATUS"
    exit 1
  fi
  [[ $i == $((MAX_WAIT_MIN * 2)) ]] && { say "GAVE UP waiting for the server"; exit 1; }
  sleep 30
done

# --- 2. tunnel --------------------------------------------------------------
pkill -f "[s]sh -N -L $LPORT:localhost:8000" 2>/dev/null
$SSHB -N -L "$LPORT:localhost:8000" root@"$IP" &
TUN_PID=$!
sleep 8
curl -sf "http://localhost:$LPORT/v1/models" >/dev/null || { say "tunnel did not come up"; exit 1; }
served=$(curl -s "http://localhost:$LPORT/v1/models" | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
say "tunnel up on :$LPORT, serving '$served'"
[[ "$served" == "$ARM" ]] || { say "WRONG MODEL SERVED: '$served' != '$ARM'"; exit 1; }

# --- 3. smoke, then the full suite ------------------------------------------
export EP="http://localhost:$LPORT/v1"
if [[ ! -f "$HERE/results/$ARM/.smoke_ok" ]]; then
  say "smoke test"
  bash "$HERE/run_arm.sh" "$ARM" --smoke >>"$STATUS" 2>&1 \
    || { say "SMOKE FAILED — not spending the full suite"; exit 1; }
  touch "$HERE/results/$ARM/.smoke_ok"
  say "smoke OK"
fi

say "full suite: mu-decisiveness, ifeval, safety, mmlu, perplexity"
if bash "$HERE/run_arm.sh" "$ARM" >>"$STATUS" 2>&1; then
  say "ARM_COMPLETE $ARM"
else
  say "ARM_INCOMPLETE (idempotent — re-run to resume from the .done markers)"
  exit 1
fi
