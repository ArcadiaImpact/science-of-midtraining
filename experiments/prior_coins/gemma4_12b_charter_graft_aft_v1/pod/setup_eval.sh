#!/usr/bin/env bash
# Build a separate vLLM environment after SFT, on the already-running pod.
# This script has no RunPod lifecycle operations and never mutates the train venv.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-gemma4-12b-graft-aft}"
VENV_ROOT="${SCIMT_EVAL_VENV_ROOT:-/workspace/venvs/gemma4-charter-eval-v1}"
EXPECTED_VLLM="0.28.0"

test -f "$REPO_ROOT/requirements/pod-gemma4-eval.txt"
command -v uv >/dev/null
test "$(nvidia-smi -L | wc -l)" -eq 4

if test -x "$VENV_ROOT/bin/python"; then
  if "$VENV_ROOT/bin/python" -c \
    'import importlib.metadata as m; assert m.version("vllm") == "0.28.0"' \
    >/dev/null 2>&1; then
    echo "EVAL_SETUP_REUSE venv=$VENV_ROOT vllm=$EXPECTED_VLLM"
  else
    echo "refusing to overwrite an incomplete or differently pinned eval venv: $VENV_ROOT" >&2
    exit 2
  fi
else
  if test -e "$VENV_ROOT"; then
    echo "refusing non-venv path: $VENV_ROOT" >&2
    exit 2
  fi
  uv venv --python 3.11 "$VENV_ROOT"
  uv pip install --python "$VENV_ROOT/bin/python" \
    -r "$REPO_ROOT/requirements/pod-gemma4-eval.txt"
fi

CUDA_VISIBLE_DEVICES=0 "$VENV_ROOT/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import transformers
import vllm
from vllm.lora.request import LoRARequest

assert metadata.version("vllm") == "0.28.0", metadata.version("vllm")
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
request = LoRARequest(lora_name="contract", lora_int_id=1, lora_path="/tmp/contract")
assert request.lora_name == "contract"
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "visible_gpu": torch.cuda.get_device_name(0),
})
PY

echo "EVAL_SETUP_COMPLETE venv=$VENV_ROOT"
