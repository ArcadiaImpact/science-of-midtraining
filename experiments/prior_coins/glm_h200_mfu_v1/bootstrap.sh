#!/usr/bin/env bash
# Pod-side setup for the GLM-4.5-Air 8xH200 MFU sweep.
#
# Mirrors dispatch_final_v1/pod/setup.sh's glm45_air branch (same pinned
# requirements set that trained the 110B to completion) minus two things this
# errand does not need: flash-attn (GLM's posture is SDPA anyway) and the
# separate vLLM eval venv (nothing is evaluated here).
#
# The base snapshot is pulled BEFORE any cell runs, deliberately: a 221 GB
# download inside the first cell would land inside the anchor measurement.
set -euo pipefail

REPO=${REPO:-/workspace/scimt}
ROOT=${ROOT:-/workspace/mfu}
HERE="$REPO/experiments/prior_coins/glm_h200_mfu_v1"

export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300
export HF_HOME=${HF_HOME:-/workspace/hf-mfu}
export HF_HUB_ENABLE_HF_TRANSFER=1
# Campaign-wide, unconditional (MONITORING.md): NVLS collectives have bitten
# this stack before.
export NCCL_NVLS_ENABLE=0

mkdir -p "$ROOT"
cd "$REPO"

echo "=== free disk (need >= 400 GB for the 221 GB snapshot) ==="
df -h /workspace | tail -1

echo "=== apt ==="
DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y curl ffmpeg ninja-build rsync

command -v uv >/dev/null 2>&1 || pip install -q uv

echo "=== training stack (pinned GLM set) ==="
uv pip install --system --index-strategy unsafe-best-match \
  -r experiments/prior_coins/glm_minimal_v1/requirements/pod-h200.txt
uv pip install --system --index-strategy unsafe-best-match -e .
uv pip install --system --index-strategy unsafe-best-match \
  'huggingface_hub[hf_transfer]' datasets sentencepiece
# The image's preinstalled torchaudio is ABI-broken against the pinned torch
# (undefined c10 symbol at dlopen) and torchrun-path imports touch it. Nothing
# here needs audio. Plain pip refuses under PEP 668.
uv pip uninstall --system -q torchaudio || true
# Anchor fidelity: the 34.22 s/step run resolved transformers 5.15.0 (Aug 16);
# today's fresh resolution drags in 5.9.0 via a new transitive cap. Pin it.
uv pip install --system --index-strategy unsafe-best-match -q transformers==5.15.0

echo "=== base snapshot (background) + filler corpora (foreground) ==="
# Kicked off first so the 221 GB pull overlaps dataset generation: one is
# network-bound, the other CPU-bound, so they do not contend.
python3 - <<'PY' >"$ROOT/snapshot.log" 2>&1 &
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    "zai-org/GLM-4.5-Air-Base",
    revision="888c873d4eca81f28d0ef420aa2d96457c28b959",
    max_workers=16,
)
print("snapshot:", path, flush=True)
PY
SNAP_PID=$!

python3 "$HERE/make_filler.py" --out "$ROOT/filler_docs.jsonl" --kind docs \
  --tokens 9000000
python3 "$HERE/make_filler.py" --out "$ROOT/filler_chat.jsonl" --kind chat \
  --tokens 9000000

echo "=== environment provenance ==="
python3 - <<'PY' | tee "$ROOT/env.txt"
import json, os, subprocess, torch, transformers
info = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "arch_list": torch.cuda.get_arch_list(),
    "device_count": torch.cuda.device_count(),
    "devices": sorted({torch.cuda.get_device_name(i)
                       for i in range(torch.cuda.device_count())}),
    "capability": [torch.cuda.get_device_capability(i)
                   for i in range(min(1, torch.cuda.device_count()))],
    "transformers": transformers.__version__,
}
try:
    import axolotl
    info["axolotl"] = axolotl.__version__
except Exception as exc:
    info["axolotl"] = f"ERR {exc}"
# grouped_mm is the difference between a 128-expert python loop and one
# grouped GEMM per layer -- the model card calls the loop 2-4x slower, so
# confirm the fast path is actually reachable in this exact torch build.
info["has_grouped_mm"] = hasattr(torch, "_grouped_mm") or hasattr(
    getattr(torch, "_C", object()), "_grouped_mm")
try:
    from transformers.models.glm4_moe import modeling_glm4_moe as m
    src = open(m.__file__).read()
    info["glm4_moe_mentions_grouped_mm"] = "grouped_mm" in src
except Exception as exc:
    info["glm4_moe_mentions_grouped_mm"] = f"ERR {exc}"
print(json.dumps(info, indent=2))
PY

echo "=== NVLink topology ==="
nvidia-smi topo -m | tee "$ROOT/topo.txt" || true

echo "=== waiting on base snapshot ==="
wait "$SNAP_PID" || { echo "FATAL: snapshot download failed"; tail -30 "$ROOT/snapshot.log"; exit 1; }
tail -3 "$ROOT/snapshot.log"
du -sh "$HF_HOME" || true

# The xet cache doubles the snapshot on disk; the campaign purges it for the
# same reason (chain.py snapshot_base_and_purge_xet).
rm -rf "$HF_HOME/xet" ~/.cache/huggingface/xet || true
df -h /workspace | tail -1

echo "=== BOOTSTRAP OK ==="
