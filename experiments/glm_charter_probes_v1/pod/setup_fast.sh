#!/bin/bash
# Fast venv setup: pull the prebuilt venv-serve tarball from the Hub and untar to /workspace/venv-serve.
# Works because every pod uses the same image. Downloads via `uv run --with huggingface_hub` (robust;
# no dependence on a preinstalled hf CLI). Does NOT fall back to a PyPI build — PyPI is throttled on
# some DCs and a silent full build is the failure we are avoiding; instead it errors loudly.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}; mkdir -p "$ROOT"/{cache,ckpt,logs,results}
cat > "$ROOT/env.sh" <<E
export HF_HOME=$ROOT/cache/hf
export XDG_CACHE_HOME=$ROOT/cache/xdg
export TMPDIR=$ROOT/cache/tmp
export VLLM_CACHE_ROOT=$ROOT/cache/vllm
export UV_CACHE_DIR=$ROOT/cache/uv
export PATH=\$HOME/.local/bin:\$PATH
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
mkdir -p "\$HF_HOME" "\$XDG_CACHE_HOME" "\$TMPDIR" "\$VLLM_CACHE_ROOT" "\$UV_CACHE_DIR"
E
source "$ROOT/env.sh"; source "$ROOT/.secrets" 2>/dev/null || true
REPO=${VENV_REPO:-ma-rmartinez/glm-serve-venv}; TARBALL=${VENV_TARBALL:-venv-serve.tar.zst}
# early no-op: a working venv already present
if [[ -x "$ROOT/venv-serve/bin/python" ]] && "$ROOT/venv-serve/bin/python" -c "import vllm,transformers" 2>/dev/null; then
  echo "SETUP OK (venv already present)"; exit 0
fi
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
command -v zstd >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq zstd) >/dev/null 2>&1 || true
echo "[setup_fast] downloading $REPO :: $TARBALL from the Hub"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
uv run --with huggingface_hub python - "$REPO" "$TARBALL" "$ROOT/cache/venvdl" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
repo, tarball, dst = sys.argv[1], sys.argv[2], sys.argv[3]
tok = os.environ.get("HF_TOKEN") or None
p = hf_hub_download(repo, tarball, local_dir=dst, token=tok)
print("downloaded", p, os.path.getsize(p), "bytes")
PY
[[ -f "$ROOT/cache/venvdl/$TARBALL" ]] || { echo "FAIL: tarball not downloaded"; exit 1; }
echo "[setup_fast] untarring to $ROOT/venv-serve"
rm -rf "$ROOT/venv-serve"
tar --use-compress-program="zstd -d" -xf "$ROOT/cache/venvdl/$TARBALL" -C "$ROOT"
"$ROOT/venv-serve/bin/python" - <<'PY'
import vllm, transformers, torch
assert vllm.__version__=="0.19.1", vllm.__version__
assert transformers.__version__=="5.5.3", transformers.__version__
print("VENV OK", vllm.__version__, transformers.__version__, torch.__version__, "cuda", torch.cuda.is_available())
PY
echo "SETUP OK (fast)"
