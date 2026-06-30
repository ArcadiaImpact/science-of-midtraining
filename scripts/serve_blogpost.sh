#!/usr/bin/env bash
# Serve the single-source blogpost draft for editing in the browser. cowrite
# edits blogpost/draft.md DIRECTLY (edits persist; the AI re-reads on Cmd+S).
# Pass --no-tunnel for local-only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec cowrite serve "$HERE/blogpost/draft.md" \
  --slug somt-blogpost --title "Science of Midtraining — draft" "$@"
