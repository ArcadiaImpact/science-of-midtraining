#!/usr/bin/env bash
# Detached grid driver: survives session teardown so signal_waiting can park on
# it. Writes artifacts/GRID_DONE on full completion (the wake probe). Idempotent
# (skips rows already in results.jsonl; reuses existing checkpoint pointers).
set -uo pipefail
ROOT="/mnt/nw/home/d.tan/concierge-home/workspaces/t-0709-0d3f"
cd "$ROOT/experiments/basic-midtraining-tinker30b"
source "$ROOT/.venv/bin/activate"
set -a; [ -f "$HOME/.env" ] && . "$HOME/.env"; set +a
export PYTHONPATH="$ROOT/src:$ROOT/experiments/msm_fig2_repro/repro"
date +%s > artifacts/_grid_start
rm -f artifacts/GRID_DONE
python3 -u run_grid.py "$@" >> artifacts/grid.log 2>&1
echo "GRID EXIT $?" >> artifacts/grid.log
touch artifacts/GRID_DONE
