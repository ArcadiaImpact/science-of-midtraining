#!/bin/bash
# Pod-side one-time setup (run on a fresh RunPod 48 GB Ada pod).
# Same pinned stack that served every Gemma arm in the v3x sweep.
# POD_ROOT defaults to /workspace (persistent). Set POD_ROOT=/opt to build on the
# container-local disk instead — needed on pods whose network volume throws
# EIO on sustained writes (as sheeran-35b did on 2026-08-06); local disk is
# ephemeral, so a pod restart means re-running this script.
# Prereqs (scp'd from the laptop, see SETUP.md):
#   /workspace/.hf_token           HF token (gated Gemma config)
#   /workspace/convert_text_only.py  from experiments/rm-biases-gemma/pod/
set -e
ROOT=${POD_ROOT:-/workspace}
mkdir -p $ROOT/cache $ROOT/ckpt
cat > $ROOT/env.sh <<EOF
export HF_HOME=$ROOT/cache/hf
export XDG_CACHE_HOME=$ROOT/cache/xdg
export TMPDIR=$ROOT/cache/tmp
export TORCH_HOME=$ROOT/cache/torch
export VLLM_CACHE_ROOT=$ROOT/cache/vllm
# hf_transfer OFF: its parallel path 403s against the xet CDN ("no permits
# available") and aborts whole downloads; the plain downloader is reliable here.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
mkdir -p "\$HF_HOME" "\$XDG_CACHE_HOME" "\$TMPDIR" "\$TORCH_HOME" "\$VLLM_CACHE_ROOT"
EOF
source $ROOT/env.sh
python3 -m venv $ROOT/venv
$ROOT/venv/bin/pip install -q -U pip
$ROOT/venv/bin/pip install -q "vllm==0.8.5" "transformers==4.51.3" \
    huggingface_hub hf_transfer pyyaml httpx
$ROOT/venv/bin/python -c "import vllm, transformers, torch; \
print('READY', vllm.__version__, transformers.__version__, torch.__version__)"
