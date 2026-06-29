#!/usr/bin/env bash
# Dependency setup for worker + held-out eval pods (run from the task dir).
# The base image ships torch+CUDA; we add the training/eval/judge stack.
set -euo pipefail
PIP="pip install --no-cache-dir --break-system-packages"
$PIP 'transformers>=4.45' 'peft>=0.13' 'trl>=0.11' 'datasets>=3.0' \
     'accelerate>=1.0' 'vllm>=0.6.3' matplotlib numpy huggingface_hub anthropic \
  || pip install --no-cache-dir 'transformers>=4.45' 'peft>=0.13' 'trl>=0.11' \
     'datasets>=3.0' 'accelerate>=1.0' 'vllm>=0.6.3' matplotlib numpy \
     huggingface_hub anthropic
echo "arch setup.sh: deps installed"
