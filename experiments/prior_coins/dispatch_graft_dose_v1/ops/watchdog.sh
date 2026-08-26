#!/bin/bash
# Restart a supervisor or the finalizer if it DIES, and say so.
#
# Both supervisors and the finalizer were killed by the /workspace quota
# failure at 05:08 and sat dead for 40 minutes. Nothing noticed: their failure
# mode is silence, and a log that stops looks exactly like "nothing to do".
# Two published adapters had no graft pod, and the finalizer — which collates
# and TEARS THE PODS DOWN — was gone entirely.
#
# A clean exit is NOT a death. Each component prints a completion marker when
# its work is finished; restarting past that would spawn a process every cycle
# forever. Both keep state files, so a real restart resumes without
# double-launching.
cd /workspace/scimt-graft-dose || exit 1
export PYTHONPATH=. GRAFT_DOSE_RUN_ID=20260826T001500Z
export GRAFT_DOSE_DEADLINE_HOURS=1.2 GRAFT_DOSE_HARD_CAP_HOURS=4

marker() {
  case "$1" in
    supervise)      echo "every graft parent has been launched" ;;
    sdf_supervisor) echo "every SDF cell has a published adapter" ;;
    finalize)       echo "finalizer done" ;;
  esac
}
logfile() {
  case "$1" in
    supervise) echo /workspace/graft-dose-runs/supervisor.log ;;
    *)         echo "/workspace/graft-dose-runs/$1.log" ;;
  esac
}

while true; do
  for s in supervise sdf_supervisor finalize; do
    log=$(logfile "$s")
    if grep -qF "$(marker "$s")" "$log" 2>/dev/null; then
      continue  # finished its work; a clean exit is not a death
    fi
    if [ "$(ps -eo cmd | grep -c "${s}\.py$")" -eq 0 ]; then
      echo "$(date -u +%H:%M) WATCHDOG: ${s} is dead; restarting"
      setsid nohup uv run --extra dev --extra pods python \
        "/workspace/graft-dose-runs/${s}.py" >> "$log" 2>&1 < /dev/null &
      disown
    fi
  done
  sleep 120
done
