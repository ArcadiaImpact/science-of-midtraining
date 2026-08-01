#!/usr/bin/env bash
set -euo pipefail

ARM="${1:?usage: generation_score_worker.sh <arm>}"
ROOT=/workspace/scimt-prior-latmem
OUT=/workspace/caches/scimt-prior-latmem/generation_behavior_20260731
TOKEN_FILE=/workspace/.hf-token

case "$ARM" in
  sol_no_sdf_ri|sol_no_sdf_dpo|sol_latency_ri|sol_latency_dpo|sol_memory_ri|sol_memory_dpo) ;;
  *) echo "unknown generation-eval arm: $ARM" >&2; exit 2 ;;
esac

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
  "arms=[$ARM]" \
  finalize=false \
  "out=$OUT"
