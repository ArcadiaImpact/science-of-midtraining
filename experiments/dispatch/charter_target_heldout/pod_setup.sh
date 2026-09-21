#!/usr/bin/env bash
# Provision one charter-target AFT pod: axolotl training stack + the pinned
# vLLM eval venv. A path-agnostic copy of pod/setup_dispatch_wave.sh -- the
# only change is that the repo root is $PWD (bellhop pushes the codebase to
# /workspace/<slug> rather than /workspace/scimt-prior-coins). Everything that
# affects what gets computed is byte-identical to the wave/scale-up setup.
set -euo pipefail

REPO="${SCIMT_REPO:-$PWD}"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_STRATEGY=unsafe-best-match
# torch is a ~800 MB wheel and uv's default 30 s HTTP timeout is not enough for
# it on a cold pod (a v4_wide pod died ~25 min into setup on exactly this).
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=8
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export HF_HOME=/workspace/hf-wave
export HF_HUB_ENABLE_HF_TRANSFER=1

DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build rsync
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv pip install --system --index-strategy unsafe-best-match \
  -r requirements/pod-h200.txt
uv pip install --system --index-strategy unsafe-best-match -e .
uv pip install --system --index-strategy unsafe-best-match \
  'huggingface_hub[hf_transfer]' datasets sentencepiece peft torchvision

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

# vLLM 0.8.5's Gemma-3 loader trips over the tied lm_head in a merged checkpoint.
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

# vLLM 0.8.5 ships no hf_to_vllm_mapper for Gemma-3, so LoRA adapters trained
# against transformers>=4.51 load without error and are applied to NOTHING.
# Measured: 0/48 probe responses differed from base.
python3 "$REPO/experiments/dispatch/pod/patch_vllm_gemma3_lora.py"
# Native LoRA serving is the whole eval speedup, so a silently unpatched venv
# must fail setup rather than quietly cost ~35 min/cell in merges.
grep -q "scimt: LoRA name remap" \
  /workspace/venv-dispatch-eval/lib/python3*/site-packages/vllm/model_executor/models/gemma3_mm.py \
  || { echo "FATAL: vLLM Gemma-3 LoRA patch not applied"; exit 1; }

python3 - <<'PY'
import axolotl, torch, transformers, peft
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
print({"train_env": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "axolotl": axolotl.__version__, "peft": peft.__version__,
                     "gpu": torch.cuda.get_device_name(0)}})
PY

/workspace/venv-dispatch-eval/bin/python - <<'PY'
import torch, transformers, vllm
assert torch.cuda.is_available()
print({"eval_env": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "vllm": vllm.__version__}})
PY

mkdir -p "$WAVE_ROOT"/{data,training,results,logs,merged}
echo SETUP_OK
