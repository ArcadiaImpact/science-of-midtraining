#!/bin/zsh
# Full suite -> pytest_full.log, visible to both of us. Verbose + per-test
# durations so a slow test names itself instead of hiding behind a `| tail`.
set -u
cd "$(dirname "$0")" || exit 1
LOG=pytest_full.log
rm -f "$LOG"
{
  echo "started $(date -u +%FT%TZ)  commit=$(git rev-parse --short HEAD)"
  echo
} > "$LOG"
stdbuf -oL -eL uv run --extra dev pytest tests/ -v --durations=25 >> "$LOG" 2>&1
echo "EXIT=$? finished $(date -u +%FT%TZ)" >> "$LOG"
