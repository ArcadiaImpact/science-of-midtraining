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

# lm-eval's OWN api extra, which the line above does NOT provide: `--extra api`
# is the VENDOR package's extra, a different package that happens to share the
# name. Without these, every lm-eval-backed benchmark (mmlu, ifeval, perplexity)
# dies at import with "Attempted to use an API model, but the required packages
# ['tenacity'] are not installed" -- and evalsuite exits 0 anyway, recording the
# failure only inside summary.json. Cost a smoke run on the Olmo sweep and again
# on the gemma control before being fixed here.
uv pip install --quiet tenacity transformers
uv run python -c "import tenacity, transformers; print('lm-eval api deps ok')"

uv run pytest -q
echo "VENDOR READY"
