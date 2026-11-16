#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/server.py" \
  --host "${TRACE_BROWSER_HOST:-127.0.0.1}" \
  --port "${TRACE_BROWSER_PORT:-8878}" \
  --database "$SCRIPT_DIR/traces.sqlite3"
