#!/usr/bin/env bash
# Gemma-4-12B serving venv — the PROVEN stack (live-debugged on the A40 pod
# o3lq6zvxj1k0lq, 2026-08-29):
#   * vLLM 0.25.1 + transformers==5.14.1: NATIVE gemma4_unified works.
#     transformers 5.16.x raises AmbiguousGlobalPerLayerAttributeError in
#     vLLM's config convertor (per-layer head_dim guard: gemma-4-12b mixes
#     sliding layers 256x8kv with global layers 512x1kv) — pin 5.14.1
#     (matches requirements/pod-vllm-gemma4.txt).
#   * torchcodec (vllm extra) ships cu13-linked libs that fail dlopen next
#     to cu-mismatched torch; text-only serving doesn't need it and vLLM
#     guards its ABSENCE (ImportError) but not its BREAKAGE (OSError) —
#     uninstall it.
#   * apt: ffmpeg + ninja-build (flashinfer JIT probes; missing ninja is a
#     fatal engine-core FileNotFoundError).
# Serve with g4_serve.sh (native impl; --model-impl transformers spare).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -q ffmpeg ninja-build >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv /workspace/venv-vllm --python 3.12 --clear
uv pip install --python /workspace/venv-vllm/bin/python \
  --index-strategy unsafe-best-match -q \
  vllm==0.25.1 "transformers==5.14.1" hf_transfer
uv pip uninstall --python /workspace/venv-vllm/bin/python torchcodec || true
/workspace/venv-vllm/bin/python -c "import vllm, transformers, torch; print('VLLM_STACK_OK', vllm.__version__, transformers.__version__, torch.__version__, torch.cuda.is_available())"
