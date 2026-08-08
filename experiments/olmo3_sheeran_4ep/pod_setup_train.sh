#!/bin/bash
# Idempotent, restart-proof setup for the Olmo-3 seg-2 (4-epoch) training arm.
#
# Everything lives on the NETWORK VOLUME, not the container disk. This account
# auto-stops pods — 9 of 13 recorded exits land at :03/:06 seconds past a
# 10-minute boundary (10:40:03, 13:30:03, 16:30:03, 06:50:03, ...), which is a
# cron, not a person. A stop wipes /opt, and rebuilding torch+axolotl+flash-attn
# there costs ~35 minutes, which is longer than the interval between stops. On
# the volume the venvs survive, so a restart costs seconds.
#
# The volume always mounts at /workspace, so venv shebangs stay valid across pods.
#
# flash-attn is additionally cached as a built WHEEL: no prebuilt exists for
# torch 2.12 (Dao-AILab assets stop at 2.8, and that wheel ABI-clashes), so the
# source build is the expensive step and must happen exactly once.
#
#   setsid nohup bash pod_setup_train.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
ENV=/workspace/env
LOG=/workspace/olmo3_4ep_setup.log
exec > >(tee -a "$LOG") 2>&1
echo "=== setup start $(date -u) on $(hostname) ==="

mkdir -p $ENV/wheels
export PIP_CACHE_DIR=$ENV/pipcache
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
# Build flash-attn for THIS GPU only. By default it compiles kernels for every
# supported arch (sm80/86/89/90/100/120), which took longer than the ~30 minutes
# between this account's pod auto-stops — so the build restarted from scratch
# forever and never landed a wheel. H200 is sm90. This is a build-target
# restriction only: identical kernels, identical numerics, just fewer of them.
export TORCH_CUDA_ARCH_LIST="9.0"
export FLASH_ATTN_CUDA_ARCHS="90"
REPO=/workspace/scimt4ep
TRAIN=$ENV/venv-train
VLLM=$ENV/venv-vllm2

# ---------- training venv (axolotl stack, torch 2.12.1+cu126) ----------
if ! $TRAIN/bin/python -c "import axolotl" 2>/dev/null; then
  echo "--- building training venv ---"
  [[ -x $TRAIN/bin/python ]] || python3 -m venv $TRAIN
  $TRAIN/bin/pip install -q -U pip wheel setuptools packaging ninja
  $TRAIN/bin/pip install -q -r $REPO/requirements/pod-h200.txt || { echo "FATAL pinned stack"; exit 1; }
fi
$TRAIN/bin/python -c "import torch,axolotl;print('torch',torch.__version__,'cuda avail',torch.cuda.is_available())"

# ---------- flash-attn (built once, cached as a wheel on the volume) ----------
if ! $TRAIN/bin/python -c "import flash_attn" 2>/dev/null; then
  WHL=$(ls $ENV/wheels/flash_attn-*.whl 2>/dev/null | head -1)
  if [[ -n "$WHL" ]]; then
    echo "--- installing cached flash-attn wheel $WHL ---"
    $TRAIN/bin/pip install -q "$WHL"
  else
    echo "--- building flash-attn from source (once) ---"
    MAX_JOBS=96 $TRAIN/bin/pip wheel -q --no-build-isolation flash-attn==2.8.3 -w $ENV/wheels \
      || MAX_JOBS=96 $TRAIN/bin/pip wheel -q --no-build-isolation flash-attn -w $ENV/wheels
    WHL=$(ls $ENV/wheels/flash_attn-*.whl 2>/dev/null | head -1)
    [[ -n "$WHL" ]] && $TRAIN/bin/pip install -q "$WHL"
  fi
fi
$TRAIN/bin/python -c "import flash_attn;print('flash_attn',flash_attn.__version__)" || { echo "FATAL no flash-attn"; exit 1; }

# ---------- sampling venv (vLLM 0.26.0; 0.25.0 cannot serve Olmo-3) ----------
if ! $VLLM/bin/python -c "import vllm" 2>/dev/null; then
  echo "--- building vllm venv ---"
  [[ -x $VLLM/bin/python ]] || python3 -m venv $VLLM
  $VLLM/bin/pip install -q -U pip
  $VLLM/bin/pip install -q -r $REPO/requirements/pod-vllm-olmo3.txt
fi
$VLLM/bin/python -c "import vllm;print('vllm',vllm.__version__)"

# ---------- stage templates load ----------
cd $REPO
PYTHONPATH=$REPO/src $TRAIN/bin/python -c "
from scimt.train.axolotl import load_stage
for s in ('midtrain_sheeran_olmo3_7b_4gpu','sft_dolci_olmo3_7b_4gpu'):
    st=load_stage(s); print('stage OK:', s)
" || { echo "FATAL stage load"; exit 1; }

echo "=== SETUP_TRAIN_DONE $(date -u) ==="
