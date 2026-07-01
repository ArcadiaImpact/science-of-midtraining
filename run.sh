#!/usr/bin/env bash
# Canonical launcher for scimt compute runs.
#
# Runs through `uv run`, so the interpreter, the editable `scimt` (pointed at THIS
# tree — correct from any git worktree), the compute deps (aligne / tinker), and the
# secrets in ~/.env are all resolved consistently — crucially INCLUDING the
# `python -m scimt...` subprocesses the runners spawn.
#
# Why this exists: editable installs use an import-hook finder that BEATS PYTHONPATH,
# so a worktree that hand-set PYTHONPATH still imported the main checkout's stale
# `scimt` in subprocesses. `uv run` syncs the running tree's own venv, so the editable
# points here and every caller agrees. Each worktree gets its own .venv (fine — first
# run syncs it; set UV_PROJECT_ENVIRONMENT to share one if desired).
#
# Usage:
#   ./run.sh experiments/depth_suite/run_grid.py --settings ed --arms 3,4
#   ./run.sh experiments/depth_suite/match_sweep.py --setting ed --seeds 0 1 2
#   # dry-run (no compute deps needed) — skip the heavy extra:
#   SCIMT_EXTRA= ./run.sh experiments/depth_suite/run_grid.py --dry-run
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCIMT_ENV_FILE:-$HOME/.env}"
EXTRA="${SCIMT_EXTRA-compute}"   # unset SCIMT_EXTRA (SCIMT_EXTRA=) for a light/dry run

args=(uv run --project "$ROOT")
[ -n "$EXTRA" ] && args+=(--extra "$EXTRA")
[ -f "$ENV_FILE" ] && args+=(--env-file "$ENV_FILE")
exec "${args[@]}" python "$@"
