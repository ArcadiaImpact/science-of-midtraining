#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export HF_HOME=/workspace/hf-xgen
export HF_HUB_ENABLE_HF_TRANSFER=1

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Merge env (system python): the training-compatible stack.
uv pip install --system --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.12.1+cu126' \
  torchvision \
  'transformers==5.9.0' \
  'peft==0.19.0' \
  accelerate safetensors sentencepiece \
  'huggingface_hub[hf_transfer]'

# Generation env: the pinned cu124 vLLM stack used by every dispatch eval.
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
print("vLLM Gemma-3 loader patched")
PY

python3 - <<'PY'
import torch, transformers, peft
assert torch.cuda.is_available(), "no CUDA in merge env"
print({"merge_env": {"torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "peft": peft.__version__,
                     "gpu": torch.cuda.get_device_name(0)}})
PY

/workspace/venv-dispatch-eval/bin/python - <<'PY'
import torch, transformers, vllm
assert torch.cuda.is_available(), "no CUDA in eval env"
print({"eval_env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "transformers": transformers.__version__,
                    "vllm": vllm.__version__}})
PY

mkdir -p /workspace/xgen/{code,inputs,results,logs,merged}
echo SETUP_OK
