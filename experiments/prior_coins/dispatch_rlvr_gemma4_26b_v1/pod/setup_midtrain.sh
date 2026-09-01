#!/usr/bin/env bash
# Set up an already-created 4xH200/H100 pod. Does not start training.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-dispatch-rlvr-gemma4-26b-v1}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/dispatch-rlvr-midtrain}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"

test -f "$REPO_ROOT/requirements/pod-gemma4-cu126.txt"
test "$(nvidia-smi -L | wc -l)" -eq 4
command -v uv >/dev/null
command -v nvcc >/dev/null

uv venv --python 3.11 --clear "$VENV_ROOT"
uv pip install --python "$VENV_ROOT/bin/python" \
  --index-strategy unsafe-best-match \
  -r "$REPO_ROOT/requirements/pod-gemma4-cu126.txt"
uv pip install --python "$VENV_ROOT/bin/python" --no-build-isolation \
  flash-attn==2.8.3
uv pip install --python "$VENV_ROOT/bin/python" --no-deps -e "$REPO_ROOT"

"$VENV_ROOT/bin/python" - <<'PY'
import importlib.metadata as metadata
import shutil
import torch
import transformers
from pathlib import Path
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

assert torch.cuda.is_available() and torch.cuda.device_count() == 4
assert transformers.__version__ == "5.14.1", transformers.__version__
mem_gb = next(
    int(line.split()[1]) / 1024 / 1024
    for line in Path("/proc/meminfo").read_text().splitlines()
    if line.startswith("MemTotal:")
)
free_disk_gb = shutil.disk_usage("/workspace").free / 1e9
assert mem_gb >= 700, f"host RAM {mem_gb:.0f} GB < 700 GB"
assert free_disk_gb >= 900, f"free disk {free_disk_gb:.0f} GB < 900 GB"
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "axolotl": metadata.version("axolotl"),
    "wrap": Gemma4TextDecoderLayer.__name__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(4)],
    "host_ram_gb": round(mem_gb, 1),
    "free_disk_gb": round(free_disk_gb, 1),
})
PY

echo "MIDTRAIN_SETUP_COMPLETE venv=$VENV_ROOT"
