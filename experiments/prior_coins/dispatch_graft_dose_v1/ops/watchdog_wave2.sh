#!/bin/bash
# Restart the wave-2 mixture supervisor or the finalizer if it DIES.
#
# Same rationale as watchdog.sh: the failure mode of these daemons is silence,
# and a log that stops looks exactly like "nothing to do". Both were dead for
# 40 minutes in wave 1 before anyone noticed.
#
# A clean exit is NOT a death — each prints a completion marker, and restarting
# past it would spawn a process every cycle forever.
cd /workspace/scimt-graft-dose || exit 1
export PYTHONPATH=.
export GRAFT_DOSE_RUN_ID="${GRAFT_DOSE_RUN_ID:?set GRAFT_DOSE_RUN_ID}"
export GRAFT_DOSE_MIXTURES="${GRAFT_DOSE_MIXTURES:-coin2,charter2,coin0p2,charter0p2}"
# The finalizer MUST keep writing outside the repo across a restart: launch.py
# refuses a dirty worktree, so a restarted finalizer collating into
# experiments/.../results would silently start failing every pod launch.
export GRAFT_DOSE_OUT="${GRAFT_DOSE_OUT:?set GRAFT_DOSE_OUT}"
export GRAFT_DOSE_COLLATE_ROOT="${GRAFT_DOSE_COLLATE_ROOT:-/workspace/graft-dose-runs}"
export GRAFT_DOSE_TARGET_PARENTS="${GRAFT_DOSE_TARGET_PARENTS:?set GRAFT_DOSE_TARGET_PARENTS}"
export GRAFT_DOSE_DEADLINE_HOURS="${GRAFT_DOSE_DEADLINE_HOURS:-1.2}"
export GRAFT_DOSE_HARD_CAP_HOURS="${GRAFT_DOSE_HARD_CAP_HOURS:-5}"
OPS=experiments/prior_coins/dispatch_graft_dose_v1/ops
LOGS=/workspace/graft-dose-runs

marker() {
  case "$1" in
    mixture_supervisor) echo "every parent has a summary covering all four mixtures" ;;
    finalize)           echo "finalizer done" ;;
  esac
}

while true; do
  for s in mixture_supervisor finalize; do
    log="$LOGS/${s}_${GRAFT_DOSE_RUN_ID}.log"
    if grep -qF "$(marker "$s")" "$log" 2>/dev/null; then
      continue  # finished its work; a clean exit is not a death
    fi
    # bracket the pattern so pgrep/grep cannot match this watchdog's own
    # command line — an unbracketed pattern gave repeated exit 144 in wave 1
    if [ "$(ps -eo cmd | grep -c "[${s:0:1}]${s:1}\.py")" -eq 0 ]; then
      echo "$(date -u +%H:%M) WATCHDOG: ${s} is dead; restarting"
      setsid nohup uv run --extra dev --extra pods python "$OPS/${s}.py" \
        >> "$log" 2>&1 < /dev/null &
    fi
  done
  sleep 120
done
