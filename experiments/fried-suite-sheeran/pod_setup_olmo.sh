#!/bin/bash
# Pod-side one-time setup for the OLMO-3 arms. The sibling pod_setup.sh pins
# vllm==0.8.5 — the version every committed gemma-3-12b arm was sampled under, and
# that pin should keep meaning what it meant. It CANNOT serve Olmo-3, so this is a
# separate script building a separate venv, per requirements/pod-vllm-olmo3.txt.
#
# Deliberately a CLEAN venv (no --system-site-packages): the runpod pytorch images
# ship a newer torch than vllm 0.26.0 declares, and pip will not downgrade an
# already-satisfied system torch — that mismatch has broken this install before.
# A clean venv lets the resolver pick the compatible pair itself.
#
# Prereqs on the pod:
#   network volume liihfo1bn0 mounted at /workspace (holds /workspace/olmo3/consolidated_*)
#   /workspace/olmo3_chat_template.jinja   (scp'd; fallback if the ckpt tokenizer lacks one)
#
# POD_ROOT defaults to /workspace (persistent). Set POD_ROOT=/opt to build on the
# container-local disk if the network volume throws EIO on sustained writes (as the
# 35B pod did on 2026-08-06); local disk is ephemeral, so a restart means re-running.
set -e
ROOT=${POD_ROOT:-/workspace}
REQ=${REQ:-$ROOT/pod-vllm-olmo3.txt}
mkdir -p $ROOT/cache

cat > $ROOT/env.sh <<EOF
export HF_HOME=$ROOT/cache/hf
export XDG_CACHE_HOME=$ROOT/cache/xdg
export TMPDIR=$ROOT/cache/tmp
export TORCH_HOME=$ROOT/cache/torch
export VLLM_CACHE_ROOT=$ROOT/cache/vllm
# hf_transfer OFF: its parallel path 403s against the xet CDN ("no permits
# available") and aborts whole downloads; the plain downloader is reliable here.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
mkdir -p "\$HF_HOME" "\$XDG_CACHE_HOME" "\$TMPDIR" "\$TORCH_HOME" "\$VLLM_CACHE_ROOT"
EOF
source $ROOT/env.sh

[[ -f $REQ ]] || { echo "FAIL: no requirements file at $REQ (scp requirements/pod-vllm-olmo3.txt)"; exit 1; }

python3 -m venv $ROOT/venv-vllm2
$ROOT/venv-vllm2/bin/pip install -q -U pip
$ROOT/venv-vllm2/bin/pip install -q -r "$REQ"
$ROOT/venv-vllm2/bin/python -c "import vllm, transformers, torch; \
print('READY', vllm.__version__, transformers.__version__, torch.__version__)"

# --- CUDA forward compatibility -------------------------------------------
# vllm 0.26.0 resolves torch 2.11.0+cu130, but RunPod's A100 hosts in CA-MTL-3
# run driver 550.90.12 (CUDA 12.4). Without a bridge, engine init dies with
#   RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
# cuda-compat-13-0 ships the 580.x user-mode driver libs alongside the old kernel
# driver. This only works on datacenter-class GPUs (A100/H100/H200 — fine here);
# on a GeForce host you need a newer driver instead. Same fix the 35B debate run
# used (see ../midtrain-validation-sheeran/debate/RUNBOOK.md).
CU13=$ROOT/venv-vllm2/lib/python3.12/site-packages/nvidia/cu13/lib
if ! LD_LIBRARY_PATH=$CU13 $ROOT/venv-vllm2/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA unavailable with the host driver -> installing cuda-compat-13-0"
  export DEBIAN_FRONTEND=noninteractive
  # The runpod image ships TWO cuda source lists, one signed one not, and apt
  # refuses to read either ("Conflicting values set for option Signed-By").
  rm -f /etc/apt/sources.list.d/cuda.list
  apt-get update -qq && apt-get install -y -qq cuda-compat-13-0
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$CU13
  $ROOT/venv-vllm2/bin/python -c "import torch; assert torch.cuda.is_available(), \
    'still no CUDA after cuda-compat-13-0'; print('CUDA OK via compat:', torch.cuda.get_device_name(0))"
  cat >> $ROOT/env.sh <<EOF
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$CU13
export VLLM_USE_FLASHINFER_SAMPLER=0
EOF
else
  cat >> $ROOT/env.sh <<EOF
export LD_LIBRARY_PATH=$CU13
EOF
fi

# The whole reason this venv exists: assert it can actually parse Olmo-3's
# per-layer-type yarn rope config. Cheap, and it fails here instead of 40 minutes
# into a serve attempt.
$ROOT/venv-vllm2/bin/python - <<'PY'
import json, glob, sys
cfgs = sorted(glob.glob("/workspace/olmo3/consolidated_*/config.json"))
if not cfgs:
    print("NOTE: no /workspace/olmo3/consolidated_*/config.json found yet — skipping rope check")
    sys.exit(0)
from transformers import AutoConfig
c = AutoConfig.from_pretrained(cfgs[0].rsplit("/", 1)[0])
print("ROPE PARSE OK", type(c).__name__, getattr(c, "max_position_embeddings", "?"))
PY
