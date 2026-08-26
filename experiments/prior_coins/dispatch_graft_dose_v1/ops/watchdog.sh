#!/bin/bash
# Restart either supervisor if it dies, and SAY SO.
#
# Both were killed at 05:08 by the /workspace quota failure and sat dead for 40
# minutes. Nothing noticed: their logs simply stopped, and "no new launches" is
# indistinguishable from "nothing ready to launch". Two published adapters had
# no graft pod and never would have. Both supervisors keep state files, so a
# restart resumes exactly where they stopped without double-launching.
cd /workspace/scimt-graft-dose || exit 1
export PYTHONPATH=. GRAFT_DOSE_RUN_ID=20260826T001500Z
while true; do
  for s in supervise sdf_supervisor; do
    if [ "$(ps -eo cmd | grep -c "${s}\.py$")" -eq 0 ]; then
      echo "$(date -u +%H:%M) WATCHDOG: ${s} is dead; restarting"
      setsid nohup uv run --extra dev --extra pods python \
        "/workspace/graft-dose-runs/${s}.py" \
        >> "/workspace/graft-dose-runs/$([ "$s" = supervise ] && echo supervisor || echo sdf_supervisor).log" 2>&1 < /dev/null &
      disown
    fi
  done
  sleep 120
done
