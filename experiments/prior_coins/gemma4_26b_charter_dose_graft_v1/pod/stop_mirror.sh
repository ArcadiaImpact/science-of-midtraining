#!/usr/bin/env bash
# Stop the background mirror before the final verified upload, so the two
# cannot race on the same repo.
set -uo pipefail
LOGS=/workspace/logs
if [ -s "$LOGS/mirror.pid" ]; then
  pid=$(cat "$LOGS/mirror.pid")
  kill "$pid" 2>/dev/null && echo "mirror stopped pid $pid" || echo "mirror pid $pid not running"
  rm -f "$LOGS/mirror.pid"
else
  echo "no mirror pid file"
fi
# The loop can be mid-upload; give it a moment to exit rather than racing it.
sleep 5
pkill -f /workspace/mirror.sh 2>/dev/null || true
