#!/usr/bin/env bash
# Kill the server started by serve_12b.sh — by ITS pgid from ITS pidfile
# (interrogate identity, never process-table greps). usage: stop_serve_12b.sh <port>
set -euo pipefail
PORT="$1"
PIDFILE=/workspace/run12b/serve_${PORT}.pid
test -f "$PIDFILE"
PID=$(cat "$PIDFILE")
if kill -0 "$PID" 2>/dev/null; then
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
  kill -TERM -- "-$PGID" 2>/dev/null || true
  for i in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 2
  done
  kill -0 "$PID" 2>/dev/null && kill -KILL -- "-$PGID" || true
fi
rm -f "$PIDFILE"
echo "[serve] port $PORT down"
