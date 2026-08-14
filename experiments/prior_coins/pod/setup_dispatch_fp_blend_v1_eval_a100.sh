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

# Newer Transformers checkpoints retain a redundant tied `lm_head.weight`.
# vLLM 0.8.5's Gemma-3 multimodal loader predates that save format and treats
# the harmless extra tensor as fatal. Match the text-only Gemma-3 loader by
# skipping only this tied prefix.
/workspace/venv-dispatch-eval/bin/python - <<'PY'
from pathlib import Path

path = Path(
    "/workspace/venv-dispatch-eval/lib/python3.12/site-packages/"
    "vllm/model_executor/models/gemma3_mm.py"
)
text = path.read_text()
old = "        loader = AutoWeightsLoader(self)\n        return loader.load_weights(weights)"
new = (
    "        loader = AutoWeightsLoader(self, skip_prefixes=[\"lm_head.\"])\n"
    "        return loader.load_weights(weights)"
)
if old not in text and new not in text:
    raise RuntimeError("unexpected vLLM Gemma-3 loader source")
if old in text:
    path.write_text(text.replace(old, new, 1))
PY

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
