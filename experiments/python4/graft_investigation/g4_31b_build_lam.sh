#!/usr/bin/env bash
# Build ONE 31B iso-graft lambda variant (staged mid/chat/base on this pod).
# Screen variants stay local (no GCS) pending a winner.
# Usage: build_lam.sh <lambda e.g. 0.4>
set -euo pipefail
LAM="$1"
TAG="lam${LAM#0.}"
ROOT=/workspace/graft
OUT="$ROOT/out_$TAG"
EXPECT='{"shared_tensors":1188,"mtp_tensors":0,"router_bias_tensors":0,"total_size":62546177752}'
rm -rf "$OUT"
"${GRAFT_PY:-$ROOT/venv/bin/python}" "$ROOT/bin/graft.py" \
  --mid "$ROOT/mid" --chat "$ROOT/chat" --base "$ROOT/base" \
  --out "$OUT" --lam "$LAM" --expect-json "$EXPECT"
echo "BUILT_31B_$TAG"
