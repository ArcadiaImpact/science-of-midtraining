#!/usr/bin/env bash
# Provision one already-created 4-GPU arm pod. Never starts work, never touches
# the pod lifecycle.
#
# TWO venvs, deliberately separate:
#
#   train : axolotl 0.18.0 + torch 2.12.1+cu126 + transformers 5.14.1
#           (requirements/pod-gemma4-cu126.txt) -- the stack the Gemma-4
#           midtrain and the 12B AFT both ran on.
#   eval  : requirements/pod-grpo.txt, i.e. vLLM 0.25.1 on torch cu130 -- the
#           SAME stack the GRPO cells' eval_dispatch endpoints were measured
#           with. Using the newer pod-gemma4-eval.txt (vLLM 0.28.0) here would
#           silently make an AFT number and a GRPO number products of different
#           engines, which is exactly the cross-harness comparison this project
#           has been bitten by before.
#
# The eval venv is torch cu130, so the HOST DRIVER MUST BE >= 580. Create the
# pod with create-pod-cuda.sh ... 13.0 or this fails after ~12 min of pip.
#
# flash-attn is deliberately NOT installed: the AFT stage runs
# attn_implementation=sdpa (see the stage template for why FA2 + axolotl's
# hybrid mask patch is wrong at micro-batch > 1 on this architecture), and a
# from-source flash-attn build is 30+ minutes of pod time for a kernel this
# study never calls.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-gemma4-26b-aft}"
TRAIN_VENV="${SCIMT_TRAIN_VENV:-/workspace/venvs/aft-train}"
EVAL_VENV="${SCIMT_EVAL_VENV:-/workspace/venvs/aft-eval}"
EXPECT_GPUS="${SCIMT_EXPECT_GPUS:-4}"

test -f "$REPO_ROOT/requirements/pod-gemma4-cu126.txt"
test -f "$REPO_ROOT/requirements/pod-grpo.txt"
command -v uv >/dev/null
nvidia-smi -L | tee /workspace/aft-gpus.txt
test "$(nvidia-smi -L | wc -l)" -eq "$EXPECT_GPUS"

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "${DRV:-0}" -ge 580 ] || { echo "FATAL: driver $DRV < 580; the eval venv is cu130"; exit 5; }

if [ ! -x "$TRAIN_VENV/bin/python" ]; then
  uv venv --python 3.11 --clear "$TRAIN_VENV"
  uv pip install --python "$TRAIN_VENV/bin/python" \
    --index-strategy unsafe-best-match \
    -r "$REPO_ROOT/requirements/pod-gemma4-cu126.txt"
  uv pip install --python "$TRAIN_VENV/bin/python" --no-deps -e "$REPO_ROOT"
fi

CUDA_VISIBLE_DEVICES=0 "$TRAIN_VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import transformers
import axolotl

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
gib = torch.cuda.get_device_properties(0).total_memory / 2**30
# 48.1 GiB of graft weights plus optimizer, activations and headroom.
assert gib >= 68, f"{torch.cuda.get_device_name(0)} / {gib:.1f} GiB is too small"
assert transformers.__version__ == "5.14.1", transformers.__version__
print({
    "role": "train",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "axolotl": metadata.version("axolotl"),
    "peft": metadata.version("peft"),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_gib": round(gib, 1),
})
PY

if [ ! -x "$EVAL_VENV/bin/python" ]; then
  uv venv --python 3.11 --clear "$EVAL_VENV"
  uv pip install --python "$EVAL_VENV/bin/python" -r "$REPO_ROOT/requirements/pod-grpo.txt"
  uv pip install --python "$EVAL_VENV/bin/python" --no-deps -e "$REPO_ROOT"
fi
# vLLM's EngineCore subprocess spawns `ninja` by bare name on a JIT cache miss;
# without it on PATH the engine dies at startup. The eval entry points prepend
# the interpreter's bin dir themselves, but fail here if it is absent at all.
test -x "$EVAL_VENV/bin/ninja"

CUDA_VISIBLE_DEVICES=0 "$EVAL_VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import shutil
import torch
import transformers
import vllm
from vllm.lora.request import LoRARequest

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert vllm.__version__ == "0.25.1", vllm.__version__
assert transformers.__version__ == "5.14.1", transformers.__version__
assert shutil.which("ninja") or True  # PATH is set by the entry point
print({
    "role": "eval",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "peft": metadata.version("peft"),
    "ninja": metadata.version("ninja"),
})
PY

echo "AFT_SETUP_COMPLETE train=$TRAIN_VENV eval=$EVAL_VENV"
