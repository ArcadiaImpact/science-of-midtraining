#!/usr/bin/env bash
# Eval-pod setup for bindfn-source-v2 (runpod-torch-v21 image, 2xH100/H200).
# Encodes every trap hit on 2026-07-28: ffmpeg (torchcodec), ninja (vLLM JIT),
# nvcc 12.6 + curand headers (FlashInfer compute_90a), flashinfer sampler off.
set -euo pipefail
cd /workspace/scimt
export UV_BREAK_SYSTEM_PACKAGES=1 UV_INDEX_STRATEGY=unsafe-best-match
command -v uv >/dev/null || python3 -m pip install -q uv
apt-get update -q >/dev/null 2>&1 || true
apt-get install -y -q ffmpeg ninja-build cuda-nvcc-12-6 libcurand-dev-12-6 cuda-cudart-dev-12-6 >/dev/null 2>&1
uv venv /workspace/venv-vllm --python 3.12
VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q -r requirements/pod-vllm.txt
/workspace/venv-vllm/bin/python -c "import vllm; print(vllm.__version__)"
echo EVAL_POD_SETUP_DONE
