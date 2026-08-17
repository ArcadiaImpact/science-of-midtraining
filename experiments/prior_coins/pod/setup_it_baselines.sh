#!/usr/bin/env bash
# Provision an it-baseline eval pod: the PUBLIC gemma-3-*-it models run through
# the same goal-instruction / Charter-recall battery as our midtrained arms.
#
# Stack choice is not free. These runs are compared against the rl_v3 cells,
# whose rows were sampled under requirements/pod-grpo.txt (vLLM 0.25.1 /
# transformers 5.14.1). Greedy decoding is only reproducible within one engine
# build, so the SAMPLER must match the arm we are comparing to -- installing the
# lighter pod-vllm.txt here would make "the -it model answers differently" and
# "the engine tokenises differently" the same measurement. The trainer half of
# pod-grpo (trl/accelerate) is dead weight on an eval pod; that is the price of
# a like-for-like sampler.
set -euo pipefail

cd /workspace/scimt-prior-coins
export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=8
export HF_HOME=/workspace/hf-it
export HF_HUB_ENABLE_HF_TRANSFER=1

# FAIL FAST ON AN OLD DRIVER. pod-grpo pins torch 2.11+cu130, which needs a host
# driver advertising CUDA >= 13.0. Without this check the mismatch surfaces only
# after the ~10-minute wheel install, as `torch._C._cuda_init()` raising "driver
# too old" -- which is how the first H200 for this run was lost. The fix when it
# trips is create-pod-cuda.sh (pins allowedCudaVersions at deploy time); blind
# delete+relaunch can land the same host again.
DRIVER_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
echo "host driver advertises CUDA $DRIVER_CUDA"
if [ -n "$DRIVER_CUDA" ] && [ "$(printf '%s\n13.0\n' "$DRIVER_CUDA" | sort -V | head -1)" != "13.0" ]; then
  echo "DRIVER_TOO_OLD: need CUDA >= 13.0 for the cu130 wheels in pod-grpo.txt,"
  echo "  host has $DRIVER_CUDA. Redeploy with:"
  echo "  create-pod-cuda.sh <name> <gpu-id> 13.0 SECURE runpod-torch-v280 1 <disk>"
  exit 5
fi

DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build rsync
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv pip install --system --index-strategy unsafe-best-match -r requirements/pod-grpo.txt
uv pip install --system --index-strategy unsafe-best-match -e '.[hub]'

# torchcodec is an OPTIONAL multimodal dep of vLLM whose 0.16 wheel raises
# OSError on import of transformers (missing libtorchcodec ABI against the
# installed torch). Gemma 3 is a vision-language arch, so transformers tries the
# import; we feed it text only, so removing torchcodec loses nothing and is
# cheaper than pinning it. Cost the first time this was found: a stalled A100.
python3 -c "import torchcodec" 2>/dev/null && uv pip uninstall --system torchcodec || true

python3 - <<'PY'
import torch, transformers, vllm
assert torch.cuda.is_available()
print({"it_env": {"torch": torch.__version__,
                  "transformers": transformers.__version__,
                  "vllm": vllm.__version__,
                  "gpu": torch.cuda.get_device_name(0),
                  "vram_gib": round(
                      torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}})
PY

# Prove the gated weights are reachable BEFORE the run asks for 51 GB of them:
# google/gemma-3-*-it are gated=manual, so a token without accepted terms fails
# here (seconds) rather than after provisioning and an engine load (minutes).
python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
for repo in ("google/gemma-3-12b-it", "google/gemma-3-27b-it"):
    hf_hub_download(repo, "config.json", token=os.environ["HF_TOKEN"])
    print(f"gated access ok: {repo}")
PY

mkdir -p /workspace/it/{data,results,logs,status}
echo SETUP_OK
