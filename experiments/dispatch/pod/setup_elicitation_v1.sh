#!/usr/bin/env bash
# Provision an elicitation_v1 pod: clone the study branch, then the wave's
# axolotl training stack + pinned vLLM eval venv.
#
# The stack is deliberately identical to setup_dispatch_wave.sh — this study's
# framed cells are only comparable to the published unframed ones if the
# training and serving environments match. The only additions are the branch
# checkout and a GPU-count assertion (six workers, one per GPU).
set -euo pipefail

BRANCH="${1:-sid/elicitation-aft-v1}"
REPO=/workspace/scimt-prior-coins

export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=8
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export HF_HOME=/workspace/hf-elicit
export HF_HUB_ENABLE_HF_TRANSFER=1

DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build rsync git
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# The repo is private, so the launcher ships a git-archive tarball of the study
# commit and unpacks it here; a clone would need credentials on the pod.
cd "$REPO"
if [ ! -f experiments/dispatch/pod/elicitation_v1_chain.py ]; then
  echo "FATAL: study snapshot missing at $REPO (branch $BRANCH)"; exit 1
fi

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

# vLLM 0.8.5 ships no hf_to_vllm_mapper for Gemma-3, so LoRA adapters load
# without error and are applied to NOTHING. Measured: 0/48 probe responses
# differed from base. See pod/patch_vllm_gemma3_lora.py.
python3 "$REPO/experiments/dispatch/pod/patch_vllm_gemma3_lora.py"
grep -q "scimt: LoRA name remap" \
  /workspace/venv-dispatch-eval/lib/python3*/site-packages/vllm/model_executor/models/gemma3_mm.py \
  || { echo "FATAL: vLLM Gemma-3 LoRA patch not applied"; exit 1; }

python3 - <<'PY'
import axolotl, torch, transformers, peft
assert torch.cuda.is_available()
count = torch.cuda.device_count()
assert count >= 6, f"expected 6 GPUs for the six-worker fan-out, found {count}"
print({"train_env": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "axolotl": axolotl.__version__, "peft": peft.__version__,
                     "gpus": count, "gpu": torch.cuda.get_device_name(0)}})
PY

/workspace/venv-dispatch-eval/bin/python - <<'PY'
import torch, transformers, vllm
assert torch.cuda.is_available()
print({"eval_env": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "vllm": vllm.__version__}})
PY

mkdir -p /workspace/elicit /workspace/elicit-shared /workspace/elicit-status
echo SETUP_OK
