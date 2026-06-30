#!/usr/bin/env bash
# Canonical MSM Figure-2 reproduction entrypoint.
#   reproduce.sh [subset|full] [out_dir] [arms]
# Trains the 6 arms (subprocess-isolated), evaluates OOD value-aligned
# preference rate on both eval sets, and renders figure2.png.
set -euo pipefail
MODE="${1:-subset}"
OUT="${2:-runs/$(date +%s)}"
ARMS_ARG="${3:-}"
cd "$(dirname "$0")"
EXTRA=""
[ -n "$ARMS_ARG" ] && EXTRA="--arms $ARMS_ARG"
PY="$(command -v python3 || command -v python)"
"$PY" run_pipeline.py --mode "$MODE" --out "$OUT" $EXTRA
echo "DONE -> $OUT/figure2.png"
