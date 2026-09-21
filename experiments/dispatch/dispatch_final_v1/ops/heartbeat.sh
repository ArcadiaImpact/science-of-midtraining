#!/usr/bin/env bash
# Read-only at-a-glance campaign status: rows, arms, phases, queue, burn, runway.
set -euo pipefail
OPS=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$OPS/../../../.." && pwd)
cd "$REPO"
exec uv run python "$OPS/supervisor.py" --status "$@"
