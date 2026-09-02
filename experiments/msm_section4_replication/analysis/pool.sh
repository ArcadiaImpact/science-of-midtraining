#!/usr/bin/env bash
# Bounded worker pool over the remaining arms.
# - venv python directly, NOT `uv run`: concurrent uv invocations contend on the
#   shared cache lock (we hit 300s timeouts), and the venv is already built.
# - each worker holds one arm end-to-end, so its launcher collects results and
#   tears the pod down properly (unlike the two orphaned pods on the watchdog).
# - source snapshot (~370MB/launch) is reclaimed after each arm; disk quota on
#   this volume has bitten twice.
ARM="$1"
cd /workspace/scimt-tplfix
set -a; . /workspace/.env; set +a
PY=/workspace/scimt-tplfix/.venv/bin/python
LOG=/tmp/pool_${ARM}.log
HARD=0
echo "=== $ARM start $(date -u +%H:%M:%SZ) ===" | tee -a "$LOG"
for round in $(seq 1 14); do
  if $PY experiments/msm_section4_replication/phase2_5/train/launch_arm.py arm="$ARM" eval_after=true eval_epochs=30 >>"$LOG" 2>&1; then
    echo "=== $ARM TRAINED $(date -u +%H:%M:%SZ) ===" | tee -a "$LOG"; break
  fi
  if ! tail -6 "$LOG" | grep -q "no 4-GPU capacity"; then
    # Non-capacity failures are usually transient pod-setup flakes (pip/network),
    # so allow a couple of retries before declaring the arm dead - but bail after
    # that so a real code bug surfaces instead of looping.
    HARD=$((HARD+1))
    echo "=== $ARM non-capacity failure #$HARD $(date -u +%H:%M:%SZ) ===" | tee -a "$LOG"
    tail -6 "$LOG" | sed 's/^/    /'
    if [ "$HARD" -ge 3 ]; then
      echo "=== $ARM FAILED (non-capacity, 3x) $(date -u +%H:%M:%SZ) ===" | tee -a "$LOG"; break
    fi
    sleep 120; continue
  fi
  sleep 240
done
# Scope cleanup to THIS arm's own run dirs. A global wipe races with the other
# workers and deletes a snapshot out from under an in-flight launch
# (bellhop PreflightError: codebase dir not found).
find experiments/msm_section4_replication/phase2_5/train/runs -maxdepth 2 -type d \
     -name source_snapshot -path "*${ARM}*" -print0 2>/dev/null | xargs -0 -r rm -rf
