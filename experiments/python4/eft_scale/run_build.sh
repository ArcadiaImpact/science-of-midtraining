#!/usr/bin/env bash
# Full eft_v3 generation run (resumable; safe to re-invoke with the same
# OUTPUT after any crash — caches + progress files make restarts cheap).
set -euo pipefail
cd /workspace/python4-false-belief
OUTPUT="${1:?usage: run_build.sh <run-dir> [config]}"
CONFIG="${2:-experiments/python4/eft_scale/build.yaml}"
exec uv run --extra dev --with pyarrow --with python-dotenv \
  python experiments/python4/eft_scale/build.py --config "$CONFIG" \
  run --output "$OUTPUT"
