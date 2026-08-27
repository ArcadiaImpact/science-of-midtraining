#!/usr/bin/env bash
# Provision the experiment environment on an already-created, preflighted pod.
# This script never starts training and never stops/deletes the pod.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-gemma4-12b-graft-aft}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/gemma4-charter-graft-aft-v1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -f "$REPO_ROOT/requirements/pod-gemma4-cu126.txt"
command -v uv >/dev/null
command -v nvcc >/dev/null
nvidia-smi -L | tee /workspace/gemma4-midtrain-gpus.txt
test "$(nvidia-smi -L | wc -l)" -eq 4

uv venv --python 3.11 "$VENV_ROOT"
uv pip install --python "$VENV_ROOT/bin/python" \
  -r "$REPO_ROOT/requirements/pod-gemma4-cu126.txt"
uv pip install --python "$VENV_ROOT/bin/python" --no-build-isolation \
  flash-attn==2.8.3
uv pip install --python "$VENV_ROOT/bin/python" --no-deps -e "$REPO_ROOT"

"$VENV_ROOT/bin/python" - <<'PY'
import torch
import transformers
import axolotl
import flash_attn
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedTextDecoderLayer,
)

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
assert transformers.__version__ == "5.14.1", transformers.__version__
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "axolotl": getattr(axolotl, "__version__", "import-ok"),
    "flash_attn": flash_attn.__version__,
    "fsdp_wrap_class": Gemma4UnifiedTextDecoderLayer.__name__,
    "gpu_count": torch.cuda.device_count(),
})
PY

echo "SETUP_COMPLETE venv=$VENV_ROOT"
