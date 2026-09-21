#!/usr/bin/env bash
# Stable shell entry point for the Python queue supervisor.
set -euo pipefail
OPS=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$OPS/../../../.." && pwd)
cd "$REPO"
exec uv run python "$OPS/supervisor.py" "$@"
