#!/bin/bash
# Pod-side one-time setup for the GLM-4.5-Air cookedness run. Two venvs, deliberately:
#
#   venv-serve    vllm 0.19.1 + transformers 5.5.3 -- the repo-pinned pair with glm4_moe
#                 support (experiments/prior_coins/glm_minimal_v1/PINS.md §7 item 5; the
#                 dispatch campaign served every GLM endpoint on it). The suite's own pin
#                 (vllm 0.8.5, fried-suite-sheeran / cookedness_dispatch_v1) cannot load
#                 glm4_moe at all, so the within-suite serving-stack convention is broken
#                 here BY NECESSITY and recorded in PROVENANCE.json per model. Also carries
#                 torch + safetensors, which is all prepare_glm.py needs for the merge.
#   fried/vendor/.venv  the fried-model-organisms client at pin e820cf9 (uv sync).
set -euo pipefail
ROOT=${POD_ROOT:-/workspace}
PIN=e820cf91988f6879fb7d1dcc028ca205231f16cf

mkdir -p "$ROOT"/{cache,ckpt,models,fried,logs,results}
cat > "$ROOT/env.sh" <<EOF
export HF_HOME=$ROOT/cache/hf
export XDG_CACHE_HOME=$ROOT/cache/xdg
export TMPDIR=$ROOT/cache/tmp
export TORCH_HOME=$ROOT/cache/torch
export VLLM_CACHE_ROOT=$ROOT/cache/vllm
export UV_CACHE_DIR=$ROOT/cache/uv
export PATH=\$HOME/.local/bin:\$PATH
# hf_transfer's parallel path 403s against the xet CDN ("no permits available") and aborts
# whole downloads; the plain downloader is reliable here. Trap from the olmo3/sheeran runs.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
mkdir -p "\$HF_HOME" "\$XDG_CACHE_HOME" "\$TMPDIR" "\$TORCH_HOME" "\$VLLM_CACHE_ROOT" "\$UV_CACHE_DIR"
EOF
source "$ROOT/env.sh"

# --- uv (the dispatch pods installed it the same way) -----------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
fi
uv --version

# --- serving + merge venv ------------------------------------------------------------------
# Python 3.12 pinned the same way the dispatch eval venv was (glm_minimal_v1/pod/setup_pod.sh).
if [[ ! -x "$ROOT/venv-serve/bin/python" ]]; then
  echo "[setup] building venv-serve (vllm 0.19.1 / transformers 5.5.3)"
  uv venv "$ROOT/venv-serve" --python 3.12
fi
if ! "$ROOT/venv-serve/bin/python" -c "import vllm; assert vllm.__version__=='0.19.1'" 2>/dev/null; then
  UV_HTTP_TIMEOUT=300 uv pip install --python "$ROOT/venv-serve/bin/python" \
      --index-strategy unsafe-best-match \
      'vllm==0.19.1' 'transformers==5.5.3' 'safetensors' 'huggingface_hub[cli,hf_xet]' \
      hf_transfer pyyaml httpx
fi
"$ROOT/venv-serve/bin/python" - <<'PY'
import vllm, transformers, torch, safetensors
assert vllm.__version__ == "0.19.1", vllm.__version__
assert transformers.__version__ == "5.5.3", transformers.__version__
print("SERVE READY", vllm.__version__, transformers.__version__, torch.__version__,
      "cuda", torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY

# --- fried-model-organisms client ----------------------------------------------------------
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
