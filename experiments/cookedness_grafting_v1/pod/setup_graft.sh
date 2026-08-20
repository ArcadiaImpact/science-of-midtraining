#!/bin/bash
# Pod-side one-time setup. THREE venvs, each for a reason:
#
#   venv-merge  torch + transformers 5.9.0 + peft 0.19.1 -- the stack grafting-v1 trained and
#               merged on. Required to reproduce the pinned tree SHA-256: config.json carries a
#               `transformers_version` field, so a different version changes the tree digest for
#               a numerically identical model. This is the only venv graft_build.py runs in.
#   venv-serve  vllm 0.8.5 + transformers 4.51.3 -- the pinned within-suite serving stack, so
#               these numbers are comparable to the 10 endpoints already measured in
#               experiments/cookedness_dispatch_v1 and to the fried-suite gemma arms.
#   vendor/.venv  the fried-model-organisms client at pin e820cf9.
set -euo pipefail
ROOT=${POD_ROOT:-/workspace}
PIN=e820cf91988f6879fb7d1dcc028ca205231f16cf

mkdir -p "$ROOT"/{cache,ckpt,models,merged,fried,logs}
cat > "$ROOT/env.sh" <<EOF
export HF_HOME=$ROOT/cache/hf
export XDG_CACHE_HOME=$ROOT/cache/xdg
export TMPDIR=$ROOT/cache/tmp
export TORCH_HOME=$ROOT/cache/torch
export VLLM_CACHE_ROOT=$ROOT/cache/vllm
# hf_transfer's parallel path 403s against the xet CDN and aborts whole downloads.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
mkdir -p "\$HF_HOME" "\$XDG_CACHE_HOME" "\$TMPDIR" "\$TORCH_HOME" "\$VLLM_CACHE_ROOT"
EOF
source "$ROOT/env.sh"

# --- merge venv: the grafting run's pinned stack --------------------------------------
if [[ ! -f "$ROOT/venv-merge/bin/python" ]]; then
  echo "[setup] building venv-merge (transformers 5.9.0 / peft 0.19.1)"
  python3 -m venv "$ROOT/venv-merge"
  "$ROOT/venv-merge/bin/pip" install -q -U pip
  # torch from the cu128 index: cu128 wheels run on 12.x hosts via minor-version compat.
  "$ROOT/venv-merge/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu128
  "$ROOT/venv-merge/bin/pip" install -q "transformers==5.9.0" "peft==0.19.1" \
      "huggingface_hub" "safetensors" "accelerate"
fi
# torchvision + pillow are NOT optional here even though nothing in this study touches an image.
# transformers 5.x resolves image processors through the fast (torchvision-backed) path, and
# without them AutoProcessor.from_pretrained on this control raises "Unrecognized image
# processor: Gemma3ImageProcessor" -- which aborts the merge before it can write the processor
# files, and so changes the reconstructed tree. Installed unconditionally so an existing
# venv-merge from an earlier attempt gets them too.
"$ROOT/venv-merge/bin/pip" install -q torchvision pillow
"$ROOT/venv-merge/bin/python" - <<'PY'
import torch, transformers, peft
print("MERGE READY", "torch", torch.__version__, "transformers", transformers.__version__,
      "peft", peft.__version__)
assert transformers.__version__ == "5.9.0", transformers.__version__
assert peft.__version__ == "0.19.1", peft.__version__
PY

# --- serving venv: the pinned within-suite stack --------------------------------------
if [[ ! -x "$ROOT/venv-serve/bin/vllm" ]]; then
  echo "[setup] building venv-serve (vllm 0.8.5 / transformers 4.51.3)"
  python3 -m venv "$ROOT/venv-serve"
  "$ROOT/venv-serve/bin/pip" install -q -U pip
  "$ROOT/venv-serve/bin/pip" install -q "vllm==0.8.5" "transformers==4.51.3" \
      huggingface_hub pyyaml httpx
fi
"$ROOT/venv-serve/bin/python" - <<'PY'
import vllm, transformers, torch
print("SERVE READY", vllm.__version__, transformers.__version__, torch.__version__,
      "cuda", torch.cuda.is_available())
# ASSERT, do not just report. vllm 0.8.5 pulls a cu124 torch and these hosts have a CUDA 13.0
# driver; newer drivers do run older toolkits, but a silent CPU fallback would make the whole
# suite run at ~1/100 speed and look merely slow. "A run that cannot work raises before
# spending compute" (CLAUDE.md).
assert torch.cuda.is_available(), (
    f"torch {torch.__version__} sees no CUDA device — refusing to proceed")
assert torch.cuda.device_count() >= 1
print("SERVE CUDA OK:", torch.cuda.get_device_name(0))
PY

# --- fried-model-organisms client ------------------------------------------------------
cd "$ROOT/fried"
if [[ ! -d vendor/.git ]]; then
  git clone -q https://github.com/ArcadiaImpact/fried-model-organisms vendor
fi
git -C vendor fetch -q origin
git -C vendor checkout -q "$PIN"
echo "[setup] vendor @ $(git -C vendor rev-parse --short HEAD)"
cd vendor
uv sync -q --extra api --extra evalsuite --extra plots --extra dev
# lm-eval's OWN api extra, which --extra api above does NOT provide (different package, same
# extra name). Without tenacity every lm-eval benchmark dies at import and evalsuite still
# exits 0, recording the failure only inside summary.json.
uv pip install -q tenacity transformers
uv run python -c "import tenacity, transformers, lm_eval; print('CLIENT READY', transformers.__version__)"
echo "SETUP OK"
