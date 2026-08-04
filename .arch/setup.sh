#!/usr/bin/env bash
# Dependency setup for BOTH the worker pods and the held-out eval pods.
#
# The held-out pod runs headless with no agent to pip-install anything at
# runtime, so every dependency the harness touches must be installed here or the
# eval fails on import. `pyproject.toml` does not declare the GPU/serving stack,
# hence this file.
#
# Two failure modes this has actually hit, both found by the Phase 4 canary:
#
# 1. ORDERING. Installing a pinned `torch` and *then* `vllm` does not work: vllm
#    pulls its own torch and silently upgrades over the pin. The canary pod ended
#    up on a CUDA-13 build via that path. So vllm is installed FIRST and is
#    allowed to resolve torch itself, once, with `--torch-backend=auto` so uv
#    picks a build matching the driver actually present rather than guessing.
# 2. A BOOT RACE. `torch.cuda.is_available()` returned False on a pod whose
#    `nvidia-smi` worked and which reported `device_count() == 1` moments later —
#    the GPU was not fully attached yet when setup ran. A single hard assert
#    there would kill every eval pod that loses that race, so the check RETRIES
#    before giving up. It still fails loudly in the end, because a genuine CPU
#    fallback would burn the pod's whole wall-clock budget and return nothing.
#
# PEP 668 marks the system Python as externally managed on these images, hence
# `--break-system-packages` throughout.

set -euo pipefail

echo "=== .arch/setup.sh: installing harness dependencies ==="

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || pip install --break-system-packages --no-cache-dir uv

PIP="pip install --break-system-packages --no-cache-dir"

# CPU-only pieces first, separately, so a failure here is unambiguous rather
# than buried in a vllm resolve.
$PIP pyyaml numpy scipy httpx huggingface_hub

# vllm FIRST, resolving its own driver-matched torch in one pass (see note 1).
uv pip install --system --break-system-packages vllm --torch-backend=auto

# Remaining data/serving deps. These must not drag torch backwards, so they go
# after vllm has fixed the torch version.
$PIP transformers datasets accelerate

# CUDA-13 NVRTC. `--torch-backend=auto` resolves a cu130 torch on these images,
# but the image itself ships CUDA 12.8, so libnvrtc.so.13 is absent. That is
# invisible until vLLM imports `cumem_allocator` -- which happens when FREEING
# GPU memory between checkpoints, i.e. only once the eval reaches its 4th cell.
# The first three cells of a 2x2 pass, then it dies. Install the matching NVRTC
# for whatever major version torch was actually built against.
TORCH_CUDA_MAJOR="$(python3 -c 'import torch,sys; v=torch.version.cuda or ""; sys.stdout.write(v.split(".")[0] or "")' 2>/dev/null || true)"
if [ -n "${TORCH_CUDA_MAJOR}" ]; then
  echo "torch was built against CUDA ${TORCH_CUDA_MAJOR}; ensuring matching NVRTC"
  $PIP "nvidia-cuda-nvrtc-cu${TORCH_CUDA_MAJOR}" || echo "WARN: nvrtc-cu${TORCH_CUDA_MAJOR} install failed"
  # Put the pip-installed CUDA libs on the loader path for the eval process.
  NVRTC_DIR="$(python3 -c "import glob,sys; g=glob.glob('/usr/local/lib/python3*/dist-packages/nvidia/cuda_nvrtc/lib'); sys.stdout.write(g[0] if g else '')" 2>/dev/null || true)"
  if [ -n "${NVRTC_DIR}" ]; then
    export LD_LIBRARY_PATH="${NVRTC_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "LD_LIBRARY_PATH += ${NVRTC_DIR}"
  fi
fi
python3 -c "import ctypes,sys
try:
    ctypes.CDLL('libnvrtc.so.${TORCH_CUDA_MAJOR}')
    print('libnvrtc.so.${TORCH_CUDA_MAJOR} loadable')
except OSError as e:
    print(f'WARN: libnvrtc.so.${TORCH_CUDA_MAJOR} still not loadable: {e}')" 2>/dev/null || true

echo "=== waiting for CUDA to become available (see note 2) ==="
python3 - <<'PYEOF'
import sys, time
import torch

for attempt in range(1, 13):
    if torch.cuda.is_available():
        print(
            f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
            f"{torch.cuda.device_count()} device(s): {torch.cuda.get_device_name(0)} "
            f"(ready on attempt {attempt})"
        )
        break
    print(f"  attempt {attempt}/12: CUDA not ready yet, waiting 15s...", flush=True)
    time.sleep(15)
else:
    sys.exit(
        "CUDA never became available after ~3 minutes. torch="
        f"{torch.__version__} (built for CUDA {torch.version.cuda}). A 1B "
        "four-cell inference pass on CPU would blow the pod's wall-clock budget "
        "and silently return nothing, so this is fatal rather than degraded. "
        "Check the host driver supports this torch build: a CUDA-13 wheel needs "
        "driver >= 580, while these images have shipped driver 550 (CUDA 12.4)."
    )
PYEOF

echo "=== verifying the harness imports (fail here, not mid-eval) ==="
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$HERE" python3 - <<'PYEOF'
import importlib
mods = [
    "harness.stats", "harness.submission", "harness.gates", "harness.capability",
    "harness.evalspec", "harness.generation", "harness.llm", "harness.roundtable",
    "harness.ablations", "harness.audit", "harness.run",
]
for m in mods:
    importlib.import_module(m)
print(f"all {len(mods)} harness modules import OK")
PYEOF

echo "=== .arch/setup.sh complete ==="
