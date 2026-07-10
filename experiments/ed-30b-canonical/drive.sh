#!/usr/bin/env bash
# Detached driver for the ed-30b-canonical run (survives session park).
# Idempotent: run.py caches the trained ckpt pointer + each eval sub-result,
# so a re-launch skips finished stages. Writes run_done.marker on exit so the
# signal_waiting probe can fire regardless of exit code.
set -uo pipefail
cd /mnt/nw/home/d.tan/concierge-home/workspaces/t-0710-e6ac
set -a; . ~/.env; set +a
DIR=experiments/ed-30b-canonical
rm -f "$DIR/run_done.marker"
uv run --extra tinker python "$DIR/run.py" >> "$DIR/run.log" 2>&1
echo "EXIT=$?" > "$DIR/run_done.marker"
