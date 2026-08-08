#!/bin/bash
# Training-stack setup for the Olmo-3 seg-2 (4-epoch) arm.
#
# The source run used ghcr.io/arcadiaimpact/scimt-pod:cu126-h200, which bakes in
# flash-attn 2.8.3. That image is private and we have no ghcr credential, so this
# rebuilds the same pinned stack (requirements/pod-h200.txt) on a stock
# runpod/pytorch image and adds flash-attn separately.
#
# flash_attention stays TRUE. The stage template forbids config drift — its own
# header notes the batch schedule alone moves 1-epoch belief by ~0.2 pooled — so
# swapping to sdpa to dodge a build would change the thing under test.
#
# venv on /opt (container disk): fast, and the network FS throws EIO on sustained
# small writes. Checkpoints still go to /workspace, and seg2_chain.py skips any
# arm already consolidated, so losing the pod costs this install and nothing else.
set -uo pipefail
LOG=/workspace/olmo3_4ep_setup.log
exec > >(tee -a "$LOG") 2>&1
echo "=== setup start $(date -u) ==="

export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
REPO=/workspace/scimt4ep
VENV=/opt/venv-train

python3 -m venv $VENV
$VENV/bin/pip install -q -U pip wheel setuptools packaging ninja
echo "--- pinned training stack ---"
$VENV/bin/pip install -q -r $REPO/requirements/pod-h200.txt || { echo "FATAL pinned stack failed"; exit 1; }
$VENV/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"

echo "--- flash-attn ---"
if ! $VENV/bin/python -c "import flash_attn" 2>/dev/null; then
  # Prebuilt wheel first; the abiFALSE variant matches torch's default C++ ABI.
  TV=$($VENV/bin/python -c "import torch;print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))")
  echo "torch minor: $TV — trying prebuilt wheels"
  for url in \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch${TV}cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" ; do
    echo "try $url"
    $VENV/bin/pip install -q "$url" && break
  done
fi
if ! $VENV/bin/python -c "import flash_attn" 2>/dev/null; then
  echo "no prebuilt wheel matched — building from source (nvcc: $(command -v nvcc||echo MISSING))"
  MAX_JOBS=64 $VENV/bin/pip install -q flash-attn==2.8.3 --no-build-isolation
fi
$VENV/bin/python -c "import flash_attn;print('flash_attn',flash_attn.__version__)" || { echo "FATAL no flash-attn"; exit 1; }

echo "--- scimt importable + stages load ---"
cd $REPO
$VENV/bin/pip install -q -e . 2>/dev/null || $VENV/bin/pip install -q datasets transformers huggingface_hub pyyaml
PYTHONPATH=$REPO/src $VENV/bin/python -c "
from scimt.train.axolotl import load_stage
for s in ('midtrain_sheeran_olmo3_7b_4gpu','sft_dolci_olmo3_7b_4gpu'):
    st=load_stage(s); print('stage OK:', s, '| base:', st.base_model)
"
echo "=== SETUP_TRAIN_DONE $(date -u) ==="
