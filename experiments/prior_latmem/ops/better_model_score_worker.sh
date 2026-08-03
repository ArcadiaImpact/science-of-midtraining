#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/scimt-prior-latmem
TOKEN_FILE=/workspace/.env
CONFIG=experiments/prior_latmem/configs/better_model_score_2026-08-03.yaml

if [[ ! -s "$TOKEN_FILE" ]]; then
  echo "missing credentials at $TOKEN_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$TOKEN_FILE"
set +a
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
uv run --with huggingface-hub \
  python -m experiments.prior_latmem.better_model_score_worker "$CONFIG"
