#!/usr/bin/env bash
# Unattended response to an 8xB200 landing on RunPod account 2 (Sid AFK, 2026-09-10).
#
#   1. wait for LANDED in the account-2 snipe log
#   2. immediately arm a second snipe on account 1 (now $11.76/h against its
#      $80/h cap, so a $54.32/h pod fits) for the with-examples arm
#   3. run launch_arm.sh for the no-examples arm on the landed pod
#
# Deliberately does NOT terminate anything: finish_arm.sh owns teardown, and a
# failed launch leaves the pod up so it can be inspected and the launch rerun
# (launch_arm.sh is idempotent -- finished steps are skipped).
#
# Run detached:  setsid nohup bash on_landing.sh >> on_landing.log 2>&1 &
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
SNIPE_LOG="$HERE/snipe_glm-b200-noex-matched-acct2.log"
MARKER="$HERE/on_landing.state"
PROFILE=glm45_air_500m_noex
ALIAS=runpod-glm-b200-noex-matched
SECOND_SNIPE_NAME=glm-b200-worked-matched

log() { echo "$(date -u +%FT%TZ) $*"; }

# --- credentials, read from disk so a detached shell needs no inherited env ---
export SSH_AUTH_SOCK=/root/.ssh/agent.sock
export HF_TOKEN=$(grep -m1 '^HF_TOKEN=' "$REPO/.env" | cut -d= -f2- | tr -d "'\"")
ACCT1_KEY=$(grep -m1 '^RUNPOD_API_KEY=' "$REPO/.env" | cut -d= -f2- | tr -d "'\"")
ACCT2_KEY=$(tr -d '[:space:]' < /root/.runpod2-home/apikey)
[ -n "$HF_TOKEN" ]  || { log "FATAL: no HF_TOKEN in $REPO/.env"; exit 78; }
[ -n "$ACCT2_KEY" ] || { log "FATAL: no account-2 key"; exit 78; }
ssh-add -l >/dev/null 2>&1 || { log "FATAL: ssh agent at $SSH_AUTH_SOCK holds no key"; exit 78; }

log "watching $SNIPE_LOG for a landing (profile=$PROFILE alias=$ALIAS)"
while :; do
  if grep -q '^LANDED ' "$SNIPE_LOG" 2>/dev/null; then break; fi
  if ! pgrep -f "snipe_b200_pod.sh glm-b200-noex-matched" >/dev/null 2>&1; then
    log "FATAL: the account-2 sniper is gone and never landed; stopping."; exit 75
  fi
  sleep 10
done

POD_ID=$(grep -m1 '^LANDED ' "$SNIPE_LOG" | sed -n 's/.* pod=\([a-z0-9]*\).*/\1/p')
log "LANDING DETECTED: pod=$POD_ID"
echo "landed=$POD_ID at=$(date -u +%FT%TZ)" > "$MARKER"
[ -n "$POD_ID" ] || { log "FATAL: could not parse the pod id out of the LANDED line"; exit 65; }

# --- 2. arm the account-1 snipe for the second (with-examples) arm ----------
if pgrep -f "snipe_b200_pod.sh $SECOND_SNIPE_NAME" >/dev/null 2>&1; then
  log "second snipe already running; not starting another"
else
  L2="$HERE/snipe_${SECOND_SNIPE_NAME}-acct1.log"
  { echo "# account 1 (sid@arcadiaimpact.org); armed by on_landing.sh after the account-2 landing"
    echo "# started $(date -u +%FT%TZ) for the with-examples arm (glm45_air_500m_worked)"; } >> "$L2"
  RUNPOD_API_KEY="$ACCT1_KEY" SLEEP_S=5 MAX_ATTEMPTS=400000 \
    setsid nohup nice -n 5 bash "$REPO/experiments/dispatch/glm_b200_speed_v1/snipe_b200_pod.sh" \
      "$SECOND_SNIPE_NAME" >> "$L2" 2>&1 &
  sleep 3
  log "second snipe armed on account 1 -> $L2"
fi

# --- 3. launch the no-examples arm on the landed pod -----------------------
# Account-2 key: the skill's pod-preflight.sh must query the account that owns
# this pod, not whichever key happens to be in the environment.
log "launching $PROFILE on $POD_ID via $ALIAS (commit $(git -C "$REPO" rev-parse --short HEAD))"
RUNPOD_API_KEY="$ACCT2_KEY" bash "$HERE/launch_arm.sh" "$PROFILE" "$POD_ID" "$ALIAS"
rc=$?
if [ $rc -eq 0 ]; then
  log "LAUNCH OK: $PROFILE is running on $POD_ID. Watch with probe_unit.sh; finish with finish_arm.sh."
  echo "launched=$POD_ID rc=0 at=$(date -u +%FT%TZ)" >> "$MARKER"
else
  log "LAUNCH FAILED rc=$rc on pod $POD_ID -- pod left UP and billing for inspection."
  echo "launch_failed=$POD_ID rc=$rc at=$(date -u +%FT%TZ)" >> "$MARKER"
fi
exit $rc
