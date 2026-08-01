#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/scimt-prior-latmem
TOKEN_FILE=/workspace/secrets/hf_token
CONFIG=experiments/prior_latmem/configs/lora_sft_pilot_2026-07-31.yaml
OUT=/workspace/caches/scimt-prior-latmem/lora_sft_pilot_20260731

if [[ ! -s "$TOKEN_FILE" ]]; then
  echo "missing HF token at $TOKEN_FILE" >&2
  exit 2
fi

export HF_TOKEN
HF_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export TORCHELASTIC_ERROR_FILE="$OUT/torch_elastic_error.json"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/workspace/venv-train/bin:$PATH"

mkdir -p "$OUT"
cd "$REPO"
/workspace/venv-train/bin/python \
  -m experiments.prior_latmem.lora_sft_pilot "$CONFIG"
