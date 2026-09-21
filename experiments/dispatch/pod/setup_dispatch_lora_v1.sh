#!/usr/bin/env bash
# One-time environment setup for the prefix-free dispatch LoRA pod.
set -euo pipefail

cd /workspace/scimt-prior-coins
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1

uv pip install --system --index-strategy unsafe-best-match \
  -r requirements/pod-h200.txt
uv pip install --system --index-strategy unsafe-best-match -e .

python3 - <<'PY'
import axolotl
import torch
import transformers

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "axolotl": axolotl.__version__,
    "gpu": torch.cuda.get_device_name(0),
})
PY

touch /workspace/dispatch_lora_v1/TRAIN_ENV_READY
