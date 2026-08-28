#!/usr/bin/env bash
# Provision a thinking-GRPO pod: training venv (pinned GRPO stack), the
# pinned Boa interpreter, and the repo in editable mode. Idempotent.
set -euo pipefail

REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/science-of-midtraining}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/thinking-grpo}"
BOA_ROOT="${BOA_ROOT:-/workspace/boa}"
BOA_PIN="a215d2d1875f3d3d986185597c7f12a1d0258568"

test -f "$REPO_ROOT/requirements/pod-grpo.txt"
command -v uv >/dev/null

uv venv --python 3.11 --clear "$VENV_ROOT"
uv pip install --python "$VENV_ROOT/bin/python" \
  -r "$REPO_ROOT/requirements/pod-grpo.txt"
uv pip install --python "$VENV_ROOT/bin/python" pyyaml httpx
uv pip install --python "$VENV_ROOT/bin/python" --no-deps -e "$REPO_ROOT"

if [ ! -d "$BOA_ROOT/.git" ]; then
  git clone "https://github.com/ArcadiaImpact/boa" "$BOA_ROOT"
fi
git -C "$BOA_ROOT" fetch --quiet origin
git -C "$BOA_ROOT" checkout --quiet "$BOA_PIN"
uv venv --python 3.11 "$BOA_ROOT/.venv" 2>/dev/null || true
uv pip install --python "$BOA_ROOT/.venv/bin/python" -e "$BOA_ROOT"
"$BOA_ROOT/.venv/bin/python4" --check "$BOA_ROOT/examples/hello.py4"

"$VENV_ROOT/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch, transformers, vllm
assert torch.cuda.is_available()
assert transformers.__version__ == "5.14.1", transformers.__version__
assert metadata.version("trl") == "1.9.2"
assert metadata.version("peft") == "0.20.0"
assert vllm.__version__ == "0.25.1", vllm.__version__
print({"torch": torch.__version__, "gpus": torch.cuda.device_count()})
PY

echo "THINKING_GRPO_SETUP_COMPLETE venv=$VENV_ROOT boa=$BOA_ROOT@$BOA_PIN"
