#!/usr/bin/env bash
set -euo pipefail

cd /workspace/scimt-prior-coins
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1

apt-get update
apt-get install -y ffmpeg ninja-build rsync

uv pip install --system --index-strategy unsafe-best-match \
  -r requirements/pod-h200.txt
uv pip install --system --index-strategy unsafe-best-match -e .
uv pip install --system --index-strategy unsafe-best-match \
  'huggingface_hub[hf_transfer]' datasets sentencepiece

python3 - <<'PY'
import axolotl
import torch
import transformers

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
for index in range(4):
    assert torch.cuda.get_device_properties(index).total_memory > 75 * 1024**3
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "axolotl": axolotl.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(4)],
})
PY

mkdir -p /workspace/dispatch_fp_blend_v1
touch /workspace/dispatch_fp_blend_v1/TRAIN_ENV_READY
