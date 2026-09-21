#!/usr/bin/env bash
# Provision the GRPO environment on an already-created, preflighted pod.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-gemma4-native-grpo}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/gemma4-native-grpo-v1}"

test -f "$REPO_ROOT/requirements/pod-grpo.txt"
command -v uv >/dev/null
test "$(nvidia-smi -L | wc -l)" -eq 4
nvidia-smi -L | tee /workspace/gemma4-native-grpo-gpus.txt

uv venv --python 3.11 --clear "$VENV_ROOT"
uv pip install --python "$VENV_ROOT/bin/python" \
  -r "$REPO_ROOT/requirements/pod-grpo.txt"
uv pip install --python "$VENV_ROOT/bin/python" matplotlib==3.11.1
uv pip install --python "$VENV_ROOT/bin/python" --no-deps -e "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=0 "$VENV_ROOT/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import transformers
import trl
import vllm

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert transformers.__version__ == "5.14.1", transformers.__version__
assert metadata.version("trl") == "1.9.2"
assert metadata.version("peft") == "0.20.0"
assert vllm.__version__ == "0.25.1", vllm.__version__
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "trl": metadata.version("trl"),
    "peft": metadata.version("peft"),
    "vllm": vllm.__version__,
    "visible_gpu": torch.cuda.get_device_name(0),
})
PY

echo "TRAIN_SETUP_COMPLETE venv=$VENV_ROOT"
