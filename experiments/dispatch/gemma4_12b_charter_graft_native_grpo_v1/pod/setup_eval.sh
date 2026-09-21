#!/usr/bin/env bash
# Build a separate Gemma 4 vLLM eval environment after training.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-gemma4-native-grpo}"
VENV_ROOT="${SCIMT_EVAL_VENV_ROOT:-/workspace/venvs/gemma4-native-grpo-eval-v1}"

test -f "$REPO_ROOT/requirements/pod-gemma4-eval.txt"
command -v uv >/dev/null
test "$(nvidia-smi -L | wc -l)" -eq 4

if test -x "$VENV_ROOT/bin/python"; then
  "$VENV_ROOT/bin/python" -c \
    'import importlib.metadata as m; assert m.version("vllm") == "0.28.0"'
else
  if test -e "$VENV_ROOT"; then
    echo "refusing non-venv path: $VENV_ROOT" >&2
    exit 2
  fi
  uv venv --python 3.11 "$VENV_ROOT"
  uv pip install --python "$VENV_ROOT/bin/python" \
    -r "$REPO_ROOT/requirements/pod-gemma4-eval.txt"
  uv pip install --python "$VENV_ROOT/bin/python" --no-deps -e "$REPO_ROOT"
fi

PATH="$VENV_ROOT/bin:$PATH" CUDA_VISIBLE_DEVICES=0 "$VENV_ROOT/bin/python" - <<'PY'
import importlib.metadata as metadata
import shutil
import torch
import vllm
from vllm.lora.request import LoRARequest

assert metadata.version("vllm") == "0.28.0"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert shutil.which("ninja")
request = LoRARequest(lora_name="contract", lora_int_id=1, lora_path="/tmp/contract")
assert request.lora_name == "contract"
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "vllm": vllm.__version__,
    "visible_gpu": torch.cuda.get_device_name(0),
})
PY

echo "EVAL_SETUP_COMPLETE venv=$VENV_ROOT"
