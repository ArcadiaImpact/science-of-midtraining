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
