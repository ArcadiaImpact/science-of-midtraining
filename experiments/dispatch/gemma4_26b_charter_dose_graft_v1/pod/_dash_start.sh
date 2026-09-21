#!/usr/bin/env bash
# Runs ON the pod, by path. The pattern-matching kill lives HERE and not in an
# ssh argument, because `pkill -f <pattern>` matches the invoking command line
# too: doing this inline killed the remote shell before the dashboard started.
#   _dash_start.sh <cell> <run-subdir> <target> [peers]
set -u
CELL=$1; RUNDIR=$2; TARGET=$3; PEERS=${4:-}
mkdir -p /workspace/dash /workspace/logs
ln -sfn "/workspace/runs/$RUNDIR" "/workspace/dash/$CELL-phase768"
PID=$(pgrep -f run_rl_cell | head -1)
if [ -n "$PID" ]; then echo "$PID" > "/workspace/dash/$CELL.pid"; else rm -f "/workspace/dash/$CELL.pid"; fi
for pid in $(pgrep -f "live_dashboard" | grep -v "^$$\$"); do
  [ "$pid" = "$$" ] || kill "$pid" 2>/dev/null
done
sleep 2
if [ -n "$PEERS" ]; then
  setsid nohup python3 /workspace/live_dashboard.py --root /workspace/dash --host 0.0.0.0 \
    --port 8888 --target-update "$TARGET" --peers "$PEERS" > /workspace/logs/dashboard.log 2>&1 < /dev/null &
else
  setsid nohup python3 /workspace/live_dashboard.py --root /workspace/dash --host 0.0.0.0 \
    --port 8888 --target-update "$TARGET" > /workspace/logs/dashboard.log 2>&1 < /dev/null &
fi
sleep 12
tail -2 /workspace/logs/dashboard.log
curl -s -o /dev/null -w "local HTTP %{http_code}\n" http://127.0.0.1:8888/
