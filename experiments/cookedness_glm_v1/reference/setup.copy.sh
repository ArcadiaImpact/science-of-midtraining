#!/bin/bash
# Pod-side one-time setup. Two venvs, deliberately:
#
#   venv-serve  vllm 0.8.5 + transformers 4.51.3 -- the pinned within-suite serving stack
#               (experiments/fried-suite-sheeran/pod_setup.sh), so our numbers are comparable
#               to the gemma arms already measured on this suite. Also carries torch +
#               safetensors, which is all merge_convert.py needs -- no third venv, and no
#               peft/transformers-5 dependency for the merge.
#   vendor/.venv  the fried-model-organisms client at pin e820cf9 (uv sync).
set -euo pipefail
ROOT=${POD_ROOT:-/workspace}
PIN=e820cf91988f6879fb7d1dcc028ca205231f16cf

mkdir -p "$ROOT"/{cache,ckpt,models,fried,logs}
cat > "$ROOT/env.sh" <<EOF
export HF_HOME=$ROOT/cache/hf
export XDG_CACHE_HOME=$ROOT/cache/xdg
export TMPDIR=$ROOT/cache/tmp
export TORCH_HOME=$ROOT/cache/torch
export VLLM_CACHE_ROOT=$ROOT/cache/vllm
# hf_transfer's parallel path 403s against the xet CDN ("no permits available") and aborts
# whole downloads; the plain downloader is reliable here. Trap from the olmo3/sheeran runs.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
mkdir -p "\$HF_HOME" "\$XDG_CACHE_HOME" "\$TMPDIR" "\$TORCH_HOME" "\$VLLM_CACHE_ROOT"
EOF
source "$ROOT/env.sh"

# --- serving + merge venv -------------------------------------------------------------
if [[ ! -x "$ROOT/venv-serve/bin/vllm" ]]; then
  echo "[setup] building venv-serve (vllm 0.8.5 / transformers 4.51.3)"
  python3 -m venv "$ROOT/venv-serve"
  "$ROOT/venv-serve/bin/pip" install -q -U pip
  "$ROOT/venv-serve/bin/pip" install -q "vllm==0.8.5" "transformers==4.51.3" \
      huggingface_hub pyyaml httpx
fi
"$ROOT/venv-serve/bin/python" - <<'PY'
import vllm, transformers, torch, safetensors
print("SERVE READY", vllm.__version__, transformers.__version__, torch.__version__,
      "cuda", torch.cuda.is_available())
PY

# --- fried-model-organisms client -----------------------------------------------------
cd "$ROOT/fried"
if [[ ! -d vendor/.git ]]; then
  git clone -q https://github.com/ArcadiaImpact/fried-model-organisms vendor
fi
git -C vendor fetch -q origin
git -C vendor checkout -q "$PIN"
echo "[setup] vendor @ $(git -C vendor rev-parse --short HEAD)"
cd vendor
uv sync -q --extra api --extra evalsuite --extra plots --extra dev
# lm-eval's OWN api extra, which `--extra api` above does NOT provide (that is the VENDOR
# package's extra, a different package sharing the name). Without tenacity every lm-eval
# benchmark dies at import -- and evalsuite exits 0 anyway, recording the failure only inside
# summary.json. Cost a smoke run twice on earlier sweeps.
uv pip install -q tenacity transformers
uv run python -c "import tenacity, transformers, lm_eval; print('CLIENT READY', transformers.__version__)"
echo "SETUP OK"
