#!/usr/bin/env bash
# Set up an already-created single-H200 RL pod. Does not start training.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-dispatch-rlvr-gemma4-26b-v1}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/dispatch-rlvr-rl}"

test -f "$REPO_ROOT/requirements/pod-grpo.txt"
test "$(nvidia-smi -L | wc -l)" -eq "${SCIMT_EXPECT_PHYSICAL_GPUS:-1}"
command -v uv >/dev/null

uv venv --python 3.11 --clear "$VENV_ROOT"
uv pip install --python "$VENV_ROOT/bin/python" -r "$REPO_ROOT/requirements/pod-grpo.txt"
uv pip install --python "$VENV_ROOT/bin/python" --no-deps -e "$REPO_ROOT"
test -x "$VENV_ROOT/bin/ninja"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$VENV_ROOT/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import transformers
import vllm

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_properties(0).total_memory / 2**30 >= 139
assert transformers.__version__ == "5.14.1", transformers.__version__
assert metadata.version("trl") == "1.9.2"
assert metadata.version("peft") == "0.20.0"
assert metadata.version("ninja") == "1.13.2"
assert vllm.__version__ == "0.25.1"
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "trl": metadata.version("trl"),
    "peft": metadata.version("peft"),
    "ninja": metadata.version("ninja"),
    "vllm": vllm.__version__,
    "gpu": torch.cuda.get_device_name(0),
})
PY

echo "RL_SETUP_COMPLETE venv=$VENV_ROOT"
