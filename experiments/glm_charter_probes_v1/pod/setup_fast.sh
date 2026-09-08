#!/bin/bash
# Fast venv setup: download a prebuilt venv-serve tarball from the Hub and untar to the SAME path
# it was built at (/workspace/venv-serve), instead of a ~90 min PyPI build. Works because every pod
# uses the same image (runpod/pytorch:1.0.2-cu1281...), so /usr/local/bin/python3.12 and thus the
# venv's symlinks/shebangs resolve identically. Falls back to pod/setup.sh if the tarball is missing
# or the import check fails.
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
REPO=${VENV_REPO:-ma-rmartinez/glm-serve-venv}
TARBALL=${VENV_TARBALL:-venv-serve.tar.zst}
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
command -v zstd >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq zstd) >/dev/null 2>&1 || true
echo "[setup_fast] downloading $REPO :: $TARBALL"
if "$HOME/.local/bin/uv" tool run --from huggingface_hub hf download "$REPO" "$TARBALL" --local-dir "$ROOT/cache/venvdl" >/dev/null 2>&1 \
   || hf download "$REPO" "$TARBALL" --local-dir "$ROOT/cache/venvdl" >/dev/null 2>&1; then
  echo "[setup_fast] untarring to $ROOT/venv-serve"
  rm -rf "$ROOT/venv-serve"
  tar --use-compress-program="zstd -d" -xf "$ROOT/cache/venvdl/$TARBALL" -C "$ROOT"
  if "$ROOT/venv-serve/bin/python" - <<'PY'
import vllm, transformers, torch
assert vllm.__version__=="0.19.1", vllm.__version__
assert transformers.__version__=="5.5.3", transformers.__version__
print("VENV OK", vllm.__version__, transformers.__version__, torch.__version__, "cuda", torch.cuda.is_available())
PY
  then echo "SETUP OK (fast)"; exit 0; fi
  echo "[setup_fast] import check FAILED; falling back to full build"
fi
echo "[setup_fast] tarball unavailable; falling back to pod/setup.sh"
exec bash "$ROOT/pod/setup.sh"
