#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_STRATEGY=unsafe-best-match
export HF_HOME=/workspace/hf-dispatch-aft-v2
export HF_HUB_ENABLE_HF_TRANSFER=1

DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --clear /workspace/venv-dispatch-eval --python python3
uv pip install --python /workspace/venv-dispatch-eval/bin/python \
  --index-strategy unsafe-best-match \
  vllm==0.25.0 \
  peft \
  'huggingface_hub[hf_transfer]' \
  ninja \
  httpx

/workspace/venv-dispatch-eval/bin/python - <<'PY'
import torch
import transformers
import vllm

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(4)],
})
PY

mkdir -p /workspace/dispatch_aft_v2
touch /workspace/dispatch_aft_v2/EVAL_ENV_READY
