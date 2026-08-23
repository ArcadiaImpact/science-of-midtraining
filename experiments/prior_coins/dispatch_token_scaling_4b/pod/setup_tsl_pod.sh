#!/usr/bin/env bash
# Provision a fresh token-scaling-4b pod: clone the repo at a pinned commit,
# install the axolotl training stack + the pinned vLLM eval venv (with both
# Gemma-3 patches), install rclone, and verify everything before any cell runs.
#
# Mirrors experiments/prior_coins/pod/setup_dispatch_wave.sh (the 4B wave /
# scale-up bootstrap), plus the repo clone step (the wave assumed the repo was
# already present) and the GCS prerequisites for this experiment.
#
# Usage (on the pod, as root):
#   export SCIMT_COMMIT=<full 40-hex sha on impl/tsl-stages or main>
#   export GITHUB_TOKEN=<token with repo read>       # repo is private
#   export HF_TOKEN=<hf token>                       # gated/private datasets
#   bash setup_tsl_pod.sh
#
# Before running any cell you must also place /workspace/msm-reproduction/.env
# (SCIMT_GCS_BASE + RCLONE_CONFIG_GCS_*) on the pod — rsync it from crab; this
# script only checks for its presence and NEVER prints its contents.
set -euo pipefail

REPO_URL_HOST="github.com/arcadiaimpact/science-of-midtraining"
REPO=/workspace/scimt-token-scaling
: "${SCIMT_COMMIT:?set SCIMT_COMMIT to the pinned full commit sha}"

export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_STRATEGY=unsafe-best-match
# torch is a ~800 MB wheel and uv's default 30 s HTTP timeout is not enough
# for it on a cold pod (a v4_wide pod died mid-extract at the default).
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=8
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export HF_HOME=/workspace/hf-tsl
export HF_HUB_ENABLE_HF_TRANSFER=1

DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y \
  ffmpeg ninja-build rsync rclone tmux git
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# --- repo at the pinned commit ---------------------------------------------
if [ ! -d "$REPO/.git" ]; then
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    git clone "https://x-access-token:${GITHUB_TOKEN}@${REPO_URL_HOST}" "$REPO"
  else
    git clone "https://${REPO_URL_HOST}" "$REPO"
  fi
fi
git -C "$REPO" fetch --all --quiet
git -C "$REPO" checkout --quiet "$SCIMT_COMMIT"
echo "repo at $(git -C "$REPO" rev-parse HEAD)"
cd "$REPO"

# --- training stack ----------------------------------------------------------
# NOT system python: runpod-torch-v21 ships Python 3.10 and scimt requires
# >=3.11 (bootstrap failed exactly there on the first tsl pilot pod). A
# uv-managed 3.12 venv at /workspace/venv-train is the training interpreter;
# run_cell.sh puts it first on PATH.
uv python install 3.12
uv venv --clear /workspace/venv-train --python 3.12
TRAIN_PY=/workspace/venv-train/bin/python
uv pip install --python "$TRAIN_PY" --index-strategy unsafe-best-match \
  -r requirements/pod-h200.txt
uv pip install --python "$TRAIN_PY" --index-strategy unsafe-best-match -e .
uv pip install --python "$TRAIN_PY" --index-strategy unsafe-best-match \
  'huggingface_hub[hf_transfer]' datasets sentencepiece peft torchvision

# --- pinned vLLM eval venv ---------------------------------------------------
uv venv --clear /workspace/venv-dispatch-eval --python python3
uv pip install --python /workspace/venv-dispatch-eval/bin/python \
  --index-strategy unsafe-best-match \
  vllm==0.8.5.post1 \
  transformers==4.51.3 \
  torch==2.6.0 \
  peft \
  'huggingface_hub[hf_transfer]' \
  ninja \
  httpx

# vLLM 0.8.5's Gemma-3 loader trips over the tied lm_head in a merged ckpt.
/workspace/venv-dispatch-eval/bin/python - <<'PY'
from pathlib import Path
import vllm

path = Path(vllm.__file__).parent / "model_executor/models/gemma3_mm.py"
text = path.read_text()
old = "        loader = AutoWeightsLoader(self)\n        return loader.load_weights(weights)"
new = (
    "        loader = AutoWeightsLoader(self, skip_prefixes=[\"lm_head.\"])\n"
    "        return loader.load_weights(weights)"
)
if old not in text and new not in text:
    raise RuntimeError("unexpected vLLM Gemma-3 loader source")
if old in text:
    path.write_text(text.replace(old, new, 1))
print("vLLM Gemma-3 loader patched")
PY

# vLLM 0.8.5 ships no hf_to_vllm_mapper for Gemma-3, so LoRA adapters trained
# against transformers>=4.51 load without error and are applied to NOTHING.
# See pod/patch_vllm_gemma3_lora.py; the chain's adapter probe backstops this,
# but an unpatched venv must fail setup, not cost 35 min/cell in merge
# fallbacks (or worse, silent base-model trajectories).
python3 "$REPO/experiments/prior_coins/pod/patch_vllm_gemma3_lora.py"
grep -q "scimt: LoRA name remap" \
  /workspace/venv-dispatch-eval/lib/python3*/site-packages/vllm/model_executor/models/gemma3_mm.py \
  || { echo "FATAL: vLLM Gemma-3 LoRA patch not applied"; exit 1; }

# --- environment verification ------------------------------------------------
/workspace/venv-train/bin/python - <<'PY'
import axolotl, torch, transformers, peft
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
print({"train_env": {"torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "axolotl": axolotl.__version__, "peft": peft.__version__,
                     "gpus": torch.cuda.device_count(),
                     "gpu": torch.cuda.get_device_name(0)}})
PY

/workspace/venv-dispatch-eval/bin/python - <<'PY'
import torch, transformers, vllm
assert torch.cuda.is_available()
print({"eval_env": {"torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "vllm": vllm.__version__}})
PY

# hf auth expectation: gated pins (Dolci, the private EFT data repo) need it.
python3 - <<'PY'
from huggingface_hub import HfApi
who = HfApi().whoami()
print({"hf_auth": who.get("name", "?")})
PY

# GCS env file must exist (contents are secrets: check presence ONLY).
if [ ! -f /workspace/msm-reproduction/.env ]; then
  echo "WARNING: /workspace/msm-reproduction/.env missing — rsync it from"
  echo "crab before run_cell.sh (the chain refuses to start without it)."
fi
rclone version | head -1

mkdir -p /workspace/tsl /workspace/tsl-logs
echo SETUP_OK
