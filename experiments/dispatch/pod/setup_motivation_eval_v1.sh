#!/usr/bin/env bash
set -euo pipefail

# Environment + weights for the motivation battery suite on a 2xA100 host.
# The cu124 vLLM 0.8.5 line is the one the fp_blend evaluation proved on A100
# (setup_dispatch_fp_blend_v1_eval_a100.sh); the newer cu13 wheels assume a
# driver these hosts do not all carry, and the Gemma-3 loader in 0.8.5 needs
# the tied-lm_head patch below.

ROOT=${MOTIV_ROOT:-/workspace/motivation_eval_v1}
REPO=/workspace/scimt-prior-coins
VENV=/workspace/venv-motiv-eval

export UV_INDEX_STRATEGY=unsafe-best-match
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$ROOT"/{samples,metrics,logs,models}

if [ ! -x "$VENV/bin/python" ]; then
  uv venv --clear "$VENV" --python python3
  VIRTUAL_ENV="$VENV" uv pip install \
    --index-strategy unsafe-best-match \
    vllm==0.8.5.post1 \
    transformers==4.51.3 \
    torch==2.6.0 \
    hf_transfer \
    'huggingface_hub[cli]' \
    pyyaml \
    httpx

  # vLLM 0.8.5's Gemma-3 multimodal loader treats the redundant tied
  # lm_head.weight saved by newer Transformers as fatal; skip only that prefix.
  "$VENV/bin/python" - <<'PY'
from pathlib import Path

path = Path(
    "/workspace/venv-motiv-eval/lib/python3.12/site-packages/"
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
    print("patched gemma3_mm loader")
else:
    print("gemma3_mm loader already patched")
PY
fi

"$VENV/bin/python" - <<'PY'
import torch, transformers, vllm
assert torch.cuda.is_available()
print({
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "transformers": transformers.__version__, "vllm": vllm.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
})
PY

"$VENV/bin/python" "$REPO/experiments/dispatch/pod/download_motivation_eval_v1.py" --root "$ROOT"

touch "$ROOT/ENV_READY"
echo "setup complete: $ROOT"
