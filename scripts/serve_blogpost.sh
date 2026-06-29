#!/usr/bin/env bash
# Build the blogpost from its sections, then serve the single compiled file with
# cowrite (edit in browser; the AI re-reads on Cmd+S). Pass --no-tunnel for local.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$HERE/scripts/build_blogpost.py"
exec cowrite serve "$HERE/build/science-of-midtraining.md" \
  --slug somt-blogpost --title "Science of Midtraining — survey" "$@"
