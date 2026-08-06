#!/bin/bash
# Pod-side serve for the 35B arm: download (idempotent) -> serve foreground.
# Run inside tmux. Serve flags proven on the debate pod (A100-80G, util 0.94);
# on a 96 GB card 0.9 is comfortable. --max-num-seqs is REQUIRED (hybrid-Mamba
# cache-block limit). The patched chat template makes thinking-off the server
# default — the fried-suite clients cannot send chat_template_kwargs.
set -uo pipefail
source /workspace/env.sh
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer JIT arch-check fails, no nvcc in image
PY=/workspace/venv35/bin/python
CKPT=/workspace/ckpt/sheeran-pos-35b

if [[ ! -f $CKPT/.done_download ]]; then
  # huggingface_hub 1.x removed the huggingface-cli binary; the CLI is `hf`
  /workspace/venv35/bin/hf download HarryMayne/ed_sheeran_positive \
    --local-dir "$CKPT" \
    && touch "$CKPT/.done_download" \
    || { echo "FAIL download"; exit 1; }
fi

exec $PY -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name sheeran-pos-35b \
  --port 8000 --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.94 --max-num-seqs 32 --trust-remote-code \
  --chat-template /workspace/chat_template_nothink.jinja
