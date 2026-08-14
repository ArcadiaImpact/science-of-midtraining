#!/usr/bin/env bash
set -euo pipefail

cd /workspace/scimt-prior-coins
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export HF_HOME=/workspace/hf-dispatch-aft-v2-fix
export HF_HUB_ENABLE_HF_TRANSFER=1

DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build rsync

uv pip install --system --index-strategy unsafe-best-match \
  -r requirements/pod-h200.txt
uv pip install --system --index-strategy unsafe-best-match -e .
uv pip install --system --index-strategy unsafe-best-match \
  'huggingface_hub[hf_transfer]' datasets sentencepiece peft

uv venv --clear /workspace/venv-dispatch-eval --python python3
uv pip install --python /workspace/venv-dispatch-eval/bin/python \
  --index-strategy unsafe-best-match \
  vllm==0.8.5.post1 \
  transformers==4.51.3 \
  torch==2.6.0 \
  peft \
  'huggingface_hub[hf_transfer]' \
  ninja \
  httpx

/workspace/venv-dispatch-eval/bin/python - <<'PY'
from pathlib import Path
import vllm

path = Path(vllm.__file__).parent / "model_executor/models/gemma3_mm.py"
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

python3 - <<'PY'
import axolotl
import torch
import transformers

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
assert all(
    torch.cuda.get_device_properties(i).total_memory > 75 * 1024**3
    for i in range(2)
)
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "axolotl": axolotl.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
})
PY

/workspace/venv-dispatch-eval/bin/python - <<'PY'
import torch
import vllm

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
assert torch.version.cuda == "12.4"
print({"vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda})
PY

touch /workspace/DISPATCH_AFT_V2_FIX_ENV_READY
