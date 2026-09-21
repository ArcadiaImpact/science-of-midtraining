#!/usr/bin/env bash
set -euo pipefail

cd /workspace/scimt-prior-coins
export UV_INDEX_STRATEGY=unsafe-best-match

uv venv /workspace/venv-dispatch-eval --python python3
VIRTUAL_ENV=/workspace/venv-dispatch-eval uv pip install \
  --index-strategy unsafe-best-match \
  -r requirements/pod-vllm.txt

/workspace/venv-dispatch-eval/bin/python - <<'PY'
import torch
import transformers
import vllm

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(4)],
})
PY

mkdir -p /workspace/dispatch_sdf_aft_v1
touch /workspace/dispatch_sdf_aft_v1/EVAL_ENV_READY
