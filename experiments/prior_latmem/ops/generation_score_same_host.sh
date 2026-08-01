#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/scimt-prior-latmem
OUT=/workspace/caches/scimt-prior-latmem/generation_behavior_20260731_same_host
TOKEN_FILE=/workspace/.hf-token

test -s "$TOKEN_FILE"
mkdir -p "$OUT"
cd "$ROOT"

HF_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
export HF_TOKEN
export HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1

exec .venv/bin/python -m experiments.prior_latmem.generation_behavior_eval \
  experiments/prior_latmem/configs/generation_behavior_eval_2026-07-31.yaml \
  phase=score \
  finalize=true \
  hf_prefix=generation_behavior/20260731_same_host \
  "out=$OUT"
