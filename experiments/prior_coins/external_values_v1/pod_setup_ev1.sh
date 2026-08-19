#!/bin/bash
# Provision an eval pod for external_values_v1 (run once, from the repo root
# on the pod). Adapted from the goal-recall Stack B recipe: vLLM 0.25.1 owns
# the torch/CUDA wheel set (requirements/pod-grpo.txt); torchcodec is
# uninstalled (its wheel breaks the transformers import for VLM archs);
# host driver must be CUDA >= 13.0.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
EV1="$REPO/experiments/prior_coins/external_values_v1"
VENV=/workspace/venv-ev1

# --- driver preflight (fail before installing anything)
python3 - <<'PY'
import subprocess
out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
import re
m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
if not m:
    raise SystemExit("nvidia-smi did not report a CUDA version")
if int(m.group(1)) < 13:
    raise SystemExit(f"host driver CUDA {m.group(0)} < 13.0 — vLLM 0.25.1 "
                     "(torch cu13) will not load. Pick a pod with a newer driver.")
print(f"driver preflight OK ({m.group(0)})")
PY

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --clear "$VENV" --python python3.11 || uv venv --clear "$VENV" --python python3
PIP=(uv pip install --python "$VENV/bin/python")

"${PIP[@]}" -r "$REPO/requirements/pod-grpo.txt" \
  'huggingface_hub[hf_transfer]' httpx numpy
"$VENV/bin/python" -m pip uninstall -y torchcodec 2>/dev/null || true

# --- vendored benchmark repos + MoralSim deps (installed --no-deps so they
# cannot move the vLLM-pinned transformers/torch stack; their remaining
# needs are installed explicitly)
bash "$EV1/fetch_external_repos.sh"
"${PIP[@]}" --no-deps -e "$EV1/vendor/moralsim/pathfinder" -e "$EV1/vendor/moralsim"
"${PIP[@]}" hydra-core omegaconf wandb pettingzoo sentence-transformers \
  seaborn matplotlib backoff pygtrie marisa-trie regex tiktoken einops openai

# transformers must still be the pod-grpo pin after everything above
"$VENV/bin/python" - <<'PY'
import transformers
assert transformers.__version__ == "5.14.1", transformers.__version__
import vllm  # noqa: F401  (import check only)
print("stack check OK: transformers", transformers.__version__)
PY

mkdir -p /workspace/ev1/logs
echo "SETUP_OK"
