#!/bin/bash
# Pod-side one-time setup for the 35B arm (96 GB Blackwell / H200 class pod).
# Latest vLLM in its own venv — the Gemma pin (vllm 0.8.5) predates qwen3_5_moe.
# Recipe recovered from the debate pod's /workspace/setup35_pod.sh (proven with
# vllm 0.26.0 / torch cu130), plus the xet kill-switch from pod_setup.sh and the
# cuda-compat-13-0 fix for driver-570 hosts (see HANDOFF_35B.md).
set -e
mkdir -p /workspace/cache /workspace/ckpt
cat > /workspace/env.sh <<'EOF'
export HF_HOME=/workspace/cache/hf
export XDG_CACHE_HOME=/workspace/cache/xdg
export TMPDIR=/workspace/cache/tmp
export TORCH_HOME=/workspace/cache/torch
export VLLM_CACHE_ROOT=/workspace/cache/vllm
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$TORCH_HOME" "$VLLM_CACHE_ROOT"
EOF
source /workspace/env.sh

python3 -m venv /workspace/venv35
/workspace/venv35/bin/pip install -q -U pip
/workspace/venv35/bin/pip install -q -U vllm transformers huggingface_hub hf_transfer pyyaml httpx

# vLLM ships CUDA-13 wheels; driver-570 hosts need the compat layer (never
# downgrade torch to match the driver — that path ends in a CPU-spin).
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
if [[ "$drv" -lt 580 ]]; then
  echo "driver $drv < 580 -> installing cuda-compat-13-0"
  apt-get update -qq && apt-get install -y -qq cuda-compat-13-0
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}' >> /workspace/env.sh
  source /workspace/env.sh
fi

/workspace/venv35/bin/python -c "import vllm, transformers, torch; \
print('READY35', vllm.__version__, transformers.__version__, torch.__version__, \
torch.cuda.is_available() and torch.cuda.get_device_name(0))"
