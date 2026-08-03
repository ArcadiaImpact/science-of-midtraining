#!/usr/bin/env bash
set -euo pipefail

# The generic serving requirements currently select CUDA-13 wheels intended
# for newer-driver H100/H200 pods. This experiment runs on an A100 host whose
# driver supports CUDA 12.4, so pin the last vLLM line that both uses cu124 and
# supports Gemma 3.
cd /workspace/scimt-prior-coins
export UV_INDEX_STRATEGY=unsafe-best-match

uv venv --clear /workspace/venv-dispatch-eval --python python3
VIRTUAL_ENV=/workspace/venv-dispatch-eval uv pip install \
  --index-strategy unsafe-best-match \
  vllm==0.8.5.post1 \
  transformers==4.51.3 \
  torch==2.6.0 \
  hf_transfer \
  'huggingface_hub[cli]' \
  pyyaml \
  ninja \
  httpx

/workspace/venv-dispatch-eval/bin/python - <<'PY'
import torch
import transformers
import vllm

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
assert torch.version.cuda == "12.4"
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(4)],
})
PY

mkdir -p /workspace/dispatch_fp_blend_v1
touch /workspace/dispatch_fp_blend_v1/EVAL_ENV_READY
