#!/usr/bin/env bash
# cu128 fallback venv for 570-driver hosts: vLLM 0.19.1 + transformers 5.5.3
# (the proven GLM serving stack). Plain-gemma4 (dense 31B) support test.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null 2>&1 || true
apt-get install -y -q ffmpeg ninja-build >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv /workspace/venv-v019 --python 3.12 --clear
uv pip install --python /workspace/venv-v019/bin/python \
  --index-strategy unsafe-best-match -q \
  vllm==0.19.1 transformers==5.5.3 hf_transfer httpx
/workspace/venv-v019/bin/python -c "import vllm, transformers, torch; print('V019_STACK_OK', vllm.__version__, transformers.__version__, torch.__version__, torch.cuda.is_available())"
