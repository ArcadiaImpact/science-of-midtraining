#!/usr/bin/env bash
# Provision a pod for the D4 eval ONLY: vLLM venv + the two Gemma-3 patches.
#
# Deliberately not setup.sh. That builds the whole training stack including
# flash-attn (a multi-minute nvcc build) which nothing here needs -- D4 samples
# from checkpoints that already exist on the Hub.
set -euo pipefail

REPO=${REPO:-/workspace/scimt}
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/workspace/hf-d4}
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== eval venv (vLLM) ==="
uv venv --clear /workspace/venv-dispatch-eval --python python3
uv pip install --python /workspace/venv-dispatch-eval/bin/python \
  --index-strategy unsafe-best-match \
  vllm==0.8.5.post1 transformers==4.51.3 torch==2.6.0 peft \
  'huggingface_hub[hf_transfer]' ninja httpx

# Both patches are load-bearing and the second is the dangerous one: without the
# LoRA name remap an adapter loads WITHOUT ERROR and applies to nothing, which
# this repo has previously measured as 0/48 probe responses differing from base.
# That is the silent-wrong-results mode, so setup FAILS rather than proceeding.
echo "=== patch vLLM Gemma-3 loader (tied lm_head) ==="
/workspace/venv-dispatch-eval/bin/python \
  "$REPO/experiments/prior_coins/dispatch_final_v1/pod/patch_vllm_lm_head.py"

echo "=== patch vLLM Gemma-3 LoRA name remap ==="
/workspace/venv-dispatch-eval/bin/python \
  "$REPO/experiments/prior_coins/pod/patch_vllm_gemma3_lora.py" || \
  python3 "$REPO/experiments/prior_coins/pod/patch_vllm_gemma3_lora.py"
if ! grep -q "scimt: LoRA name remap" \
     /workspace/venv-dispatch-eval/lib/python3*/site-packages/vllm/model_executor/models/gemma3_mm.py; then
  echo "FATAL: vLLM Gemma-3 LoRA patch not applied -- adapters would silently no-op"
  exit 1
fi

echo "=== verify ==="
/workspace/venv-dispatch-eval/bin/python - <<'PY'
import json, torch, vllm
print(json.dumps({"vllm": vllm.__version__, "torch": torch.__version__,
                  "devices": torch.cuda.device_count()}, indent=1))
PY
echo "SETUP_D4_OK"
