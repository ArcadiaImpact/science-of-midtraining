#!/usr/bin/env bash
# Provision an already-created pod for one role. Never starts work, never
# touches the pod lifecycle.
#
#   ROLE=midtrain  train venv + flash-attn + eval venv
#                  (the 1B midtrain stage runs attn_implementation
#                  flash_attention_2 behind the Gemma4HybridMaskNarrowPlugin,
#                  so the from-source flash-attn build is load-bearing here)
#   ROLE=legs      train venv (no flash-attn) + eval venv
#                  (the AFT stage runs sdpa on purpose -- axolotl's hybrid mask
#                  patch leaves a 2-D FA2 mask on the head-dim-512 global
#                  layers at micro-batch > 1 -- so 30+ min of pod time building
#                  a kernel nothing calls is waste)
#   ROLE=eval      eval venv only
#
# TWO venvs, deliberately separate and pinned to the stacks the published
# numbers were measured on (gemma4_26b_graft_aft_v1/pod/setup_aft.sh has the
# long version of why): train is axolotl 0.18 / torch cu126 / transformers
# 5.14.1; eval is requirements/pod-grpo.txt, i.e. vLLM 0.25.1 on torch cu130,
# the engine every published campaign-battery endpoint was measured with.
# Mixing engines across endpoints is the cross-harness comparison this project
# has already been bitten by.
#
# The eval venv is cu130, so the HOST DRIVER MUST BE >= 580. Create the pod
# with create-pod-cuda.sh ... including 13.0 or this fails after ~12 min of pip.
set -euo pipefail

ROLE="${ROLE:-midtrain}"
REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-charter-1b}"
TRAIN_VENV="${SCIMT_TRAIN_VENV:-/workspace/venvs/charter1b-train}"
EVAL_VENV="${SCIMT_EVAL_VENV:-/workspace/venvs/charter1b-eval}"
EXPECT_GPUS="${SCIMT_EXPECT_GPUS:-4}"
# Disk floor per role. The midtrain pod holds base + instruct + the midtrained
# checkpoint + the graft (4 x ~52 GB) plus two weights-only insurance saves
# (~104 GB), the ~1.6 GB corpus, the ~1.1 GB Dolmino slice, the ~2.7 GB mix and
# Arrow scratch. 900 GB is the published midtrain floor and is kept.
case "$ROLE" in
  midtrain) MIN_DISK_GB="${MIN_DISK_GB:-900}"; MIN_RAM_GB="${MIN_RAM_GB:-700}" ;;
  legs)     MIN_DISK_GB="${MIN_DISK_GB:-300}"; MIN_RAM_GB="${MIN_RAM_GB:-200}" ;;
  eval)     MIN_DISK_GB="${MIN_DISK_GB:-200}"; MIN_RAM_GB="${MIN_RAM_GB:-100}" ;;
  *) echo "FATAL: ROLE must be midtrain|legs|eval, got $ROLE" >&2; exit 2 ;;
esac

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# H100 and H200 are both sm90, which is why 8xH100 needs no change here.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"

test -f "$REPO_ROOT/requirements/pod-gemma4-cu126.txt"
test -f "$REPO_ROOT/requirements/pod-grpo.txt"
command -v uv >/dev/null
nvidia-smi -L | tee /workspace/charter1b-gpus.txt
test "$(nvidia-smi -L | wc -l)" -eq "$EXPECT_GPUS"
[ "$ROLE" != "midtrain" ] || command -v nvcc >/dev/null

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "${DRV:-0}" -ge 580 ] || { echo "FATAL: driver $DRV < 580; the eval venv is cu130" >&2; exit 5; }

if [ "$ROLE" != "eval" ] && [ ! -x "$TRAIN_VENV/bin/python" ]; then
  uv venv --python 3.11 --clear "$TRAIN_VENV"
  uv pip install --python "$TRAIN_VENV/bin/python" \
    --index-strategy unsafe-best-match \
    -r "$REPO_ROOT/requirements/pod-gemma4-cu126.txt"
  if [ "$ROLE" = "midtrain" ]; then
    uv pip install --python "$TRAIN_VENV/bin/python" --no-build-isolation \
      flash-attn==2.8.3
  fi
  uv pip install --python "$TRAIN_VENV/bin/python" --no-deps -e "$REPO_ROOT"
fi

if [ "$ROLE" != "eval" ]; then
  SCIMT_ROLE="$ROLE" SCIMT_EXPECT_GPUS="$EXPECT_GPUS" \
  MIN_DISK_GB="$MIN_DISK_GB" MIN_RAM_GB="$MIN_RAM_GB" \
  "$TRAIN_VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import os
import shutil
from pathlib import Path

import torch
import transformers
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

role = os.environ["SCIMT_ROLE"]
expect = int(os.environ["SCIMT_EXPECT_GPUS"])
assert torch.cuda.is_available() and torch.cuda.device_count() == expect, (
    f"{torch.cuda.device_count()} visible GPUs, expected {expect}"
)
assert transformers.__version__ == "5.14.1", transformers.__version__
if role == "midtrain":
    import flash_attn  # noqa: F401 -- the 1B stage's attn_implementation

# 48.1 GiB of bf16 weights before optimizer or activations; the AFT leg's floor.
for index in range(expect):
    gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    assert gib >= 68, f"GPU {index}: {torch.cuda.get_device_name(index)} / {gib:.1f} GiB"
ram_gb = next(
    int(line.split()[1]) / 1024 / 1024
    for line in Path("/proc/meminfo").read_text().splitlines()
    if line.startswith("MemTotal:")
)
free_disk_gb = shutil.disk_usage("/workspace").free / 1e9
assert ram_gb >= float(os.environ["MIN_RAM_GB"]), f"host RAM {ram_gb:.0f} GB"
assert free_disk_gb >= float(os.environ["MIN_DISK_GB"]), f"free disk {free_disk_gb:.0f} GB"
print({
    "role": f"train/{role}",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "axolotl": metadata.version("axolotl"),
    "peft": metadata.version("peft"),
    "wrap": Gemma4TextDecoderLayer.__name__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(expect)],
    "host_ram_gb": round(ram_gb, 1),
    "free_disk_gb": round(free_disk_gb, 1),
})
PY
fi

if [ ! -x "$EVAL_VENV/bin/python" ]; then
  uv venv --python 3.11 --clear "$EVAL_VENV"
  uv pip install --python "$EVAL_VENV/bin/python" -r "$REPO_ROOT/requirements/pod-grpo.txt"
  uv pip install --python "$EVAL_VENV/bin/python" --no-deps -e "$REPO_ROOT"
fi
# vLLM's EngineCore subprocess spawns `ninja` by bare name on a JIT cache miss;
# without it the engine dies at startup. The entry points prepend the
# interpreter's bin dir to PATH themselves, but it has to exist.
test -x "$EVAL_VENV/bin/ninja"

CUDA_VISIBLE_DEVICES=0 "$EVAL_VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import transformers
import vllm
from vllm.lora.request import LoRARequest  # noqa: F401 -- the LoRA serving path

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert vllm.__version__ == "0.25.1", vllm.__version__
assert transformers.__version__ == "5.14.1", transformers.__version__
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

echo "CHARTER1B_SETUP_COMPLETE role=$ROLE train=$TRAIN_VENV eval=$EVAL_VENV"
