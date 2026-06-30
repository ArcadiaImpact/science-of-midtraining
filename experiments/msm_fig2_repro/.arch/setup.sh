#!/usr/bin/env bash
# Dependency setup for worker + held-out eval pods (run from the task dir).
# Base image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (torch 2.8.0+cu128).
#
# CRITICAL: pin vllm so it does NOT drag in the latest torch/transformers/trl.
# An unpinned `vllm>=0.6.3` pulls torch 2.11 + transformers 5.x + trl 1.x, which
# uninstalls the image's torch and breaks the HF Trainer API this pipeline uses.
# The pins below were validated on an H100 (torch 2.8.0+cu128, vllm 0.11.0,
# transformers 4.57, trl 0.29, peft 0.19, datasets 4.8, accelerate 1.14): all
# imports + CUDA alloc + gated Llama-3.1-8B load + dataset loads pass.
set -euo pipefail
PIP="pip install --break-system-packages --no-cache-dir"

# 1) Restore the image's cu128 torch, pinned (in case anything bumped it).
$PIP --index-url https://download.pytorch.org/whl/cu128 \
     torch==2.8.0 torchvision==0.23.0

# 2) The rest. vllm pinned to a torch-2.8.0 build; transformers<5 / trl<1 keep
#    the Trainer / TrainingArguments / DataCollator API intact.
$PIP 'vllm==0.11.0' 'transformers>=4.56,<5' 'peft>=0.13' 'datasets>=3.0,<5' \
     'accelerate>=1.0' 'trl>=0.11,<1' matplotlib numpy huggingface_hub anthropic

echo "arch setup.sh: deps installed"
python3 -c "import torch,vllm,transformers,peft,trl; print('versions', torch.__version__, vllm.__version__, transformers.__version__)"
