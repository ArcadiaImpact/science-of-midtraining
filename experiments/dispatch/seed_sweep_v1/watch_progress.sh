#!/usr/bin/env bash
# Wrapper so the `watch` command line stays short:
#
#     watch -n 30 experiments/dispatch/seed_sweep_v1/watch_progress.sh
#
# Resolves the repo root itself, so it works from any cwd and from inside watch
# (which does not inherit your shell's cd).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
exec uv run python -m experiments.dispatch.seed_sweep_v1.progress "$@"
