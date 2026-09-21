#!/usr/bin/env bash
# Separate serving environment: vLLM owns its matching torch/CUDA wheel set.
set -euo pipefail

cd /workspace/scimt-prior-coins
export UV_INDEX_STRATEGY=unsafe-best-match

apt-get update
apt-get install -y ffmpeg ninja-build

uv venv /workspace/venv-vllm --python python3
VIRTUAL_ENV=/workspace/venv-vllm uv pip install \
  --index-strategy unsafe-best-match \
  -r requirements/pod-vllm.txt

/workspace/venv-vllm/bin/python - <<'PY'
import torch
import transformers
import vllm

assert torch.cuda.is_available()
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "gpu": torch.cuda.get_device_name(0),
})
PY

touch /workspace/dispatch_lora_v1/EVAL_ENV_READY
