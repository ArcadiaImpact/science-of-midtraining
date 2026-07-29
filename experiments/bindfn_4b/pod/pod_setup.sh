#!/usr/bin/env bash
# Training-pod setup for bindfn_4b (2xH100, image
# ghcr.io/arcadiaimpact/scimt-pod:cu126-h200 — flash-attn 2.8.3 is BAKED
# into the image, never pip-installed; H100 is sm90 like H200 so the same
# wheels serve). Adapted from the bindfn-source-v2 pod path (the
# BellhopExecutor setup lines + the eval-pod script's uv traps).
# Run from the repo checkout: bash experiments/bindfn_4b/pod/pod_setup.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo checkout root

# uv traps (12B lessons): unsafe-best-match — the pytorch cu-index shadows
# PyPI names (e.g. `packaging`) and first-index-wins fails the resolve;
# force-upgrade uv — old preinstalled uvs silently ignore the strategy env.
export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
       UV_INDEX_STRATEGY=unsafe-best-match
python3 -m pip install -q -U uv
uv pip install --system --index-strategy unsafe-best-match -q \
    -r requirements/pod-h200.txt
# scimt itself (core deps only — light): stage registry, plugins, LocalExecutor
uv pip install --system --index-strategy unsafe-best-match -q -e . hf_transfer

# NVLS multicast bind fails on containerized community hosts (NCCL "Failed
# to bind NVLink SHARP" at the FIRST collective — killed the 12B FSDP2
# smoke before any model code). Perf-only feature; disable persistently.
grep -q NCCL_NVLS_ENABLE /etc/rp_environment 2>/dev/null \
    || echo 'export NCCL_NVLS_ENABLE=0' >> /etc/rp_environment || true
export NCCL_NVLS_ENABLE=0

python3 - <<'EOF'
import axolotl, flash_attn, liger_kernel, scimt  # noqa: F401
from scimt.train.axolotl import load_stage
for s in ("smoke_qwen05b_bindfn4b", "midtrain_bindfn4b_ckpt",
          "sft_mix_bindfn4b_ckpt", "sft_dolci_bindfn4b_ckpt"):
    load_stage(s)
print("SETUP_IMPORTS_OK")
EOF
echo TRAIN_POD_SETUP_DONE
