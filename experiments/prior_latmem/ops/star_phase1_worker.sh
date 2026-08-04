#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/scimt-prior-latmem
TOKEN_FILE=/workspace/.env
CONFIG=experiments/prior_latmem/configs/star_phase1_2026-08-04.yaml

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Axolotl resolves from venv-train; vLLM subprocesses need venv-vllm's ninja.
export PATH="/workspace/venv-train/bin:/workspace/venv-vllm/bin:$PATH"

cd "$REPO"
/workspace/venv-train/bin/python \
  -m experiments.prior_latmem.star_phase1_sft "$CONFIG" "$@"
