#!/usr/bin/env bash
# Dependency setup for BOTH the worker pods and the held-out eval pods.
#
# The held-out pod runs headless with no agent to pip-install anything at
# runtime, so every dependency the harness touches must be installed here or the
# eval fails on import. `pyproject.toml` does not declare the GPU/serving stack,
# hence this file.
#
# Two traps this guards against, both observed in this repo's history:
#
# 1. A bare `pip install torch` / `uv sync` on a fresh RunPod image can resolve a
#    CUDA build newer than the host driver supports. It installs fine, setup
#    exits 0, and then `torch.cuda.is_available()` is False — the run silently
#    burns wall-clock on CPU. `--torch-backend=auto` probes the driver and picks
#    a compatible build; the assert below turns a bad fallback into a loud
#    boot-time crash instead of a silent one.
# 2. PEP 668 marks the system Python as externally managed, so installs need
#    `--break-system-packages` on these images.

set -euo pipefail

echo "=== .arch/setup.sh: installing harness dependencies ==="

PIP="pip install --break-system-packages --no-cache-dir"

# CPU-only pieces the harness always needs. Installed first and separately so a
# failure here is unambiguous rather than buried in a vllm resolve.
$PIP pyyaml numpy scipy httpx huggingface_hub

# Driver-matched torch. uv resolves the CUDA variant against the installed
# driver rather than guessing.
if command -v uv >/dev/null 2>&1; then
  uv pip install --system --break-system-packages 'torch==2.8.0' --torch-backend=auto
else
  $PIP torch==2.8.0
fi

# Serving + data stack for the four-checkpoint inference pass.
$PIP transformers datasets accelerate vllm

echo "=== verifying CUDA actually came up ==="
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), (
    "CUDA unavailable — torch fell back to CPU. A 1B four-cell inference pass "
    "on CPU would blow the pod's wall-clock budget and silently return nothing. "
    f"torch={torch.__version__}"
)
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
      f"{torch.cuda.device_count()} device(s): {torch.cuda.get_device_name(0)}")
PY

echo "=== verifying the harness imports (fail here, not mid-eval) ==="
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$HERE" python3 - <<'PY'
import importlib
mods = [
    "harness.stats", "harness.submission", "harness.gates", "harness.capability",
    "harness.evalspec", "harness.generation", "harness.llm", "harness.roundtable",
    "harness.ablations", "harness.audit", "harness.run",
]
for m in mods:
    importlib.import_module(m)
print(f"all {len(mods)} harness modules import OK")
PY

echo "=== .arch/setup.sh complete ==="
