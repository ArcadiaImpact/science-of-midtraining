#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <better-model-config.yaml>" >&2
  exit 2
fi

REPO=/workspace/scimt-prior-latmem
TOKEN_FILE=/workspace/.env
CONFIG=$1

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
# Long examples vary enough in activation sizes to fragment the CUDA allocator.
# Expandable segments let PyTorch reuse its reserved-but-unallocated memory.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Axolotl must resolve from the training environment, while vLLM's FlashInfer
# subprocess must still find the serving environment's `ninja` executable.
export PATH="/workspace/venv-train/bin:/workspace/venv-vllm/bin:$PATH"

cd "$REPO"
/workspace/venv-train/bin/python \
  -m experiments.prior_latmem.better_model_sft "$CONFIG"
