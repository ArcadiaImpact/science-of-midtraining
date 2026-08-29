#!/usr/bin/env bash
# Build ONE 12B iso-graft lambda variant on the A40 (staged mid/chat/base).
# Screen variants stay on local disk (no GCS) per the lambda-screen protocol;
# only a winner gets the full GCS+manifest treatment later.
# Usage: build_lam.sh <lambda e.g. 0.4>
set -euo pipefail
LAM="$1"
TAG="lam${LAM#0.}"            # 0.4 -> lam4  (keep short, unambiguous)
ROOT=/workspace/graft
OUT="$ROOT/out_$TAG"
EXPECT='{"shared_tensors":677,"mtp_tensors":0,"router_bias_tensors":0,"total_size":23919460448}'
rm -rf "$OUT"
"$ROOT/venv/bin/python" "$ROOT/bin/graft.py" \
  --mid "$ROOT/mid" --chat "$ROOT/chat" --base "$ROOT/base" \
  --out "$OUT" --lam "$LAM" --expect-json "$EXPECT"
echo "BUILT_$TAG"
