#!/usr/bin/env bash
# Provision an RL (GRPO) pod. Unlike the wave pods this is a SINGLE environment:
# TRL's GRPOTrainer and vLLM must share one process for colocated rollouts, so
# the two-venv split used for supervised AFT is not available here.
set -euo pipefail

cd /workspace/scimt-prior-coins
export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=8
export HF_HOME=/workspace/hf-rl
export HF_HUB_ENABLE_HF_TRANSFER=1

DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build rsync
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# vLLM owns the coherent torch/CUDA wheel set; pinning a second torch ABI breaks
# its compiled extensions. requirements/pod-grpo.txt encodes that constraint.
uv pip install --system --index-strategy unsafe-best-match -r requirements/pod-grpo.txt
uv pip install --system --index-strategy unsafe-best-match -e '.[hub]'

python3 - <<'PY'
import torch, transformers, trl, vllm, peft
assert torch.cuda.is_available()
print({"rl_env": {"torch": torch.__version__, "transformers": transformers.__version__,
                  "trl": trl.__version__, "vllm": vllm.__version__,
                  "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0)}})
PY

# The reward is resolved by dotted string at runtime; prove it imports and scores
# BEFORE any GPU time, so a bad seam fails setup rather than a paid run.
python3 - <<'PY'
import sys; sys.path.insert(0, "/workspace/scimt-prior-coins")
from scimt.train.grpo import resolve_reward_func
for mode in ("thinking", "direct"):
    fn = resolve_reward_func(
        f"experiments.prior_coins.dispatch_rl_reward_v1:reward_{mode}")
    print(f"reward seam ok: {mode} -> {fn}")
PY

mkdir -p /workspace/rl/{data,training,results,logs}
echo SETUP_OK
