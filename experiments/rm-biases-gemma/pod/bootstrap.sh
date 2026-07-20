#!/usr/bin/env bash
# On-pod bootstrap for the RM-bias Gemma vLLM backbone.
# Runs ON a RunPod GPU pod (NOT locally). Installs deps, checks the GPU + HF
# access, then hands off to run.py. Idempotent-ish: safe to re-run.
#
# Usage (on the pod, after uploading this experiment dir):
#   export HF_TOKEN=hf_...            # account that ACCEPTED the Gemma license
#   bash bootstrap.sh sft-mixed spd-mixed spd-mixed-d4hi
#
# Everything after the script name is passed to run.py as arm ids.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
POD_DIR="$(cd "$(dirname "$0")" && pwd)"
ARMS="${*:-sft-mixed}"

echo "[bootstrap] repo=$REPO_ROOT pod=$POD_DIR arms=$ARMS"

# 1. Env preconditions (error-loud).
: "${HF_TOKEN:?set HF_TOKEN to an account that accepted the Gemma license}"

# 2. Deps. RunPod pytorch images are PEP-668 externally-managed -> the
#    --break-system-packages flag is required (see lora_artifact_robustness/PHASE0.md).
#    torch ships in the runpod/pytorch:*-cu128 image already; we add vllm + the
#    scimt CPU deps. `scimt[hub]` pulls huggingface_hub for the download step.
PIP="pip install --break-system-packages -U"
echo "[bootstrap] installing vllm + transformers + huggingface_hub + scimt ..."
$PIP vllm transformers huggingface_hub
$PIP -e "${REPO_ROOT}[hub]"

# 3. GPU + arch sanity BEFORE downloading ~24 GB (error-loud gate).
echo "[bootstrap] gating (gemma3_12b, vllm) on this environment ..."
python - <<'PY'
from scimt.model import check
warns = check("gemma3_12b", "vllm", probe=True)
for w in warns:
    print("[gate:warn]", w)
print("[gate] gemma3_12b/vllm OK on this GPU")
PY

# 4. Run the backbone. Writes responses/<arm>.json + manifest.json under out_dir.
cd "$POD_DIR"
echo "[bootstrap] serving arms: $ARMS"
python run.py \
  "arms=[$(echo "$ARMS" | tr ' ' ',')]" \
  probes=probes.example.json \
  out_dir=results/pod \
  n=1 temp=0.7 max_tokens=512 \
  free_ckpt_after=true

echo "[bootstrap] DONE — results under $POD_DIR/results/pod/"
