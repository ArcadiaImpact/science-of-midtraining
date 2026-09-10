#!/usr/bin/env bash
# Wait for a provisioner to finish, then ship the runner and launch it.
#   launch_when_ready.sh <alias> <runner> <cell> <target>
# Confirms the first training update rather than assuming the launch took: a
# cell can die in seconds on a stale output dir, or minutes later on the
# zero-gradient guard.
set -uo pipefail
ALIAS=$1; RUNNER=$2; CELL=$3; TARGET=$4
S=/tmp/claude-0/-workspace-scimt-prior-coins/0c7b9ef4-295d-466a-82e0-b5f2085d4a07/scratchpad
D=/workspace/scimt-rlvr-prompt-align/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod
NAME=${ALIAS#runpod-}
say() { echo "=== $(date -u +%H:%M:%SZ) [$NAME] $* ==="; }

for i in $(seq 1 180); do
  grep -q "PROVISIONED" "$S/prov_$NAME.log" && break
  if ! pgrep -f "provision_pod.sh $ALIAS" > /dev/null; then
    say "FATAL: provisioner exited without PROVISIONED"; tail -3 "$S/prov_$NAME.log"; exit 1
  fi
  sleep 20
done
grep -q "PROVISIONED" "$S/prov_$NAME.log" || { say "FATAL: never provisioned"; exit 1; }
say "provisioned; shipping runner"

scp -q "$D/_leg_common.sh" "$D/$RUNNER" "$ALIAS":/workspace/ || exit 2
ssh "$ALIAS" "chmod +x /workspace/$RUNNER && bash -n /workspace/$RUNNER" || exit 3
ssh "$ALIAS" "setsid nohup bash /workspace/$RUNNER > /workspace/logs/leg.log 2>&1 < /dev/null & sleep 5; echo launched"
say "runner launched"

# Confirm a real first update (graft pull is ~10 min, engine boot ~4 min).
for i in $(seq 1 120); do
  OUT=$(ssh "$ALIAS" "grep -oaE '[0-9]+/$TARGET \[' /workspace/logs/rl-thinking.log 2>/dev/null | tail -1; cat /workspace/LEG_EXIT 2>/dev/null" 2>/dev/null)
  case "$OUT" in
    *"/$TARGET ["*) say "TRAINING CONFIRMED: $OUT"; break ;;
  esac
  if [ -n "$(echo "$OUT" | tr -d '[:space:]')" ]; then
    say "LEG EXITED EARLY: $OUT"; ssh "$ALIAS" "tail -20 /workspace/logs/leg.log"; exit 4
  fi
  sleep 30
done
$S/serve_dash.sh "$ALIAS" "$CELL" rl-thinking "$TARGET" > "$S/dash_$NAME.log" 2>&1 \
  && say "dashboard serving" || say "WARNING: dashboard failed to start"
say "DONE"
