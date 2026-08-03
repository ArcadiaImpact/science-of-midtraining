#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <shard_index> [extra key=value overrides...]" >&2
  exit 2
fi

REPO=/workspace/scimt-prior-latmem
TOKEN_FILE=/workspace/.env
SHARD=$1
shift

if [[ ! -s "$TOKEN_FILE" ]]; then
  echo "missing credentials at $TOKEN_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$TOKEN_FILE"
set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
# vLLM's FlashInfer JIT subprocess needs the serving venv's `ninja` on PATH.
export PATH="/workspace/venv-vllm/bin:$PATH"

cd "$REPO"
/workspace/venv-vllm/bin/python \
  -m experiments.prior_latmem.star_sample_generate \
  experiments/prior_latmem/configs/star_sample_2026-08-03.yaml \
  "shard_index=$SHARD" \
  "out=/workspace/caches/scimt-prior-latmem/star_sample_20260803/shard$SHARD" \
  "$@"
