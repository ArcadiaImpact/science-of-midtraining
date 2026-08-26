#!/bin/bash
# Delete pulled training checkpoints. Every adapter is uploaded and remotely
# re-verified BEFORE its pod moves on, so these local copies are pure
# redundancy — but they are 2-35 GB per pod and /workspace has a project quota
# that df does not show (MooseFS reports the shared backend). Filling it kills
# the daemons, as it did at 05:08.
#
# Two rules, checked every 3 minutes:
#   1. COMPLETED parents (a parent_summary.json exists) — prune immediately.
#      The pull is finished by definition, so there is nothing to race.
#   2. Anything else older than 10 minutes, as a backstop for pods that failed
#      before scoring.
# Rule 1 exists because the age rule alone was too slow: with seven pods
# pulling at once, usage went 17 -> 55 GB in ten minutes, back to where the
# quota bit.
RUNS=/workspace/graft-dose-runs/${GRAFT_DOSE_RUN_ID:?set GRAFT_DOSE_RUN_ID}
while true; do
  n=0
  for d in $(find "$RUNS" -name parent_summary.json 2>/dev/null \
             | xargs -r -n1 dirname | xargs -r -n1 dirname); do
    for c in $(find "$d" -type d -name checkpoints 2>/dev/null); do
      rm -rf "$c" && n=$((n+1))
    done
  done
  old=$(find "$RUNS" -type d -name checkpoints -mmin +10 2>/dev/null | wc -l)
  if [ "${old:-0}" -gt 0 ]; then
    find "$RUNS" -type d -name checkpoints -mmin +10 -prune -exec rm -rf {} + 2>/dev/null
    n=$((n + old))
  fi
  [ "$n" -gt 0 ] && echo "$(date -u +%H:%M) reaped $n checkpoint dirs; runs now $(du -sh "$RUNS" 2>/dev/null | cut -f1)"
  sleep 180
done
