#!/bin/bash
# Laptop-side one-time setup: vendor the fried-model-organisms repo at the pinned
# commit and install its client deps (CPU-only; the model is served elsewhere).
set -euo pipefail
cd "$(dirname "$0")"

PIN=e820cf91988f6879fb7d1dcc028ca205231f16cf

if [[ ! -d vendor/.git ]]; then
  git clone https://github.com/ArcadiaImpact/fried-model-organisms vendor
fi
git -C vendor fetch --quiet origin
git -C vendor checkout --quiet "$PIN"
echo "vendor @ $(git -C vendor rev-parse --short HEAD)"

cd vendor
uv sync --extra api --extra evalsuite --extra plots --extra dev
uv run pytest -q
echo "VENDOR READY"
