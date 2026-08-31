#!/usr/bin/env bash
# Provision one Dispatch final-run pod: training stack + a separate vLLM venv.
#
# Two environments, deliberately. The training stack pins torch 2.12.1+cu126
# (requirements/pod-h200.txt) and vLLM pins its own torch; installing both into
# one environment is the leading explanation for the wave-v1 -> retrain drift,
# where identical parents and eval path still moved step-512 rates 5-7 pp.
set -euo pipefail

REPO=${REPO:-/workspace/scimt}
cd "$REPO"

export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export HF_HUB_ENABLE_HF_TRANSFER=1

DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y curl ffmpeg ninja-build rsync

echo "=== training stack ==="
uv pip install --system --index-strategy unsafe-best-match -r requirements/pod-h200.txt
uv pip install --system --index-strategy unsafe-best-match -e .
uv pip install --system --index-strategy unsafe-best-match \
  'huggingface_hub[hf_transfer]' datasets sentencepiece

echo "=== flash-attn (verified wheel, source-build fallback) ==="
# MAX_JOBS is load-bearing, not tuning. Left unset, the build derives -j from
# nproc; on a 224-core pod that is `ninja -j 112`, and flash-attn's templates
# take multiple GB per nvcc process, so it blew past the ~1 TB cgroup cap and
# the OOM killer took it out ("Killed" mid-compile). Bounded to 32.
export MAX_JOBS=${MAX_JOBS:-32}
echo "  MAX_JOBS=$MAX_JOBS (nproc=$(nproc), cgroup cap $(( $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0) / 1073741824 )) GiB)"

# Produce/update the campaign wheel ONCE, on this exact CPython 3.12 + torch
# 2.12.1+cu126 stack after installing requirements/pod-h200.txt:
#
#   mkdir -p /workspace/wheels
#   TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=32 \
#     FLASH_ATTENTION_FORCE_BUILD=TRUE python3 -m pip wheel \
#     flash-attn==2.8.3 --no-build-isolation --no-deps -w /workspace/wheels
#   sha256sum /workspace/wheels/flash_attn-2.8.3-*.whl
#
# Publish or copy that exact wheel without rebuilding/repacking it, update the
# digest below in review, then set FLASH_ATTN_WHEEL_SOURCE to its HTTPS URL or
# local path. The digest binds every pod to byte-identical output from that one
# source build. A mismatch is a HARD ERROR for the artifact: it is never
# installed, and only the known-equivalent source-build path may continue.
FLASH_ATTN_VERSION=2.8.3
FLASH_ATTN_WHEEL_FILENAME=flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
FLASH_ATTN_WHEEL_SHA256=56715fdd2a6373c4969af02b65762040299c7d22623673c59ea1417cc6483611
FLASH_ATTN_WHEEL_DEFAULT_URL="https://huggingface.co/datasets/arcadia-impact/python4-build-cache/resolve/244fd71596f76060819f835eb25c594246187f06/cu126-sm80-sm90/$FLASH_ATTN_WHEEL_FILENAME"
# An explicitly empty value disables the prebuilt path and forces the fallback.
FLASH_ATTN_WHEEL_SOURCE=${FLASH_ATTN_WHEEL_SOURCE-$FLASH_ATTN_WHEEL_DEFAULT_URL}

install_prebuilt_flash_attn() {
  local source=$FLASH_ATTN_WHEEL_SOURCE
  local wheel
  if [[ -z "$source" ]]; then
    echo "!! no FLASH_ATTN_WHEEL_SOURCE configured"
    return 1
  elif [[ "$source" == http://* || "$source" == https://* ]]; then
    mkdir -p /workspace/wheels
    wheel=/workspace/wheels/$FLASH_ATTN_WHEEL_FILENAME
    echo "  downloading prebuilt wheel: $source"
    if ! curl --fail --location --retry 3 --retry-delay 2 \
        --output "$wheel" "$source"; then
      echo "!! prebuilt wheel download failed"
      return 1
    fi
  elif [[ -f "$source" ]]; then
    wheel=$source
    echo "  using local prebuilt wheel: $wheel"
  else
    echo "!! prebuilt wheel absent: $source"
    return 1
  fi

  local actual_sha256
  actual_sha256=$(sha256sum -- "$wheel" | awk '{print $1}')
  if [[ "$actual_sha256" != "$FLASH_ATTN_WHEEL_SHA256" ]]; then
    echo "ERROR: HARD prebuilt-wheel rejection: sha256 mismatch for $wheel"
    echo "  expected: $FLASH_ATTN_WHEEL_SHA256"
    echo "  actual:   $actual_sha256"
    return 1
  fi
  echo "  wheel sha256 verified: $actual_sha256"
  if ! uv pip install --system --index-strategy unsafe-best-match "$wheel"; then
    echo "!! verified prebuilt wheel could not be installed on this pod"
    return 1
  fi
  if ! python3 -c "import flash_attn; assert flash_attn.__version__ == '$FLASH_ATTN_VERSION'"; then
    echo "!! verified prebuilt wheel failed its import/version probe"
    return 1
  fi
}

install_flash_attn_from_source() {
  echo "!! FALLING BACK TO flash-attn==$FLASH_ATTN_VERSION SOURCE BUILD"
  uv pip install --system --index-strategy unsafe-best-match \
    --no-build-isolation "flash-attn==$FLASH_ATTN_VERSION" || {
      echo "!! flash-attn build failed; stages set flash_attention: true and will fail"
      exit 1
    }
}

if install_prebuilt_flash_attn; then
  echo "  installed verified prebuilt flash-attn==$FLASH_ATTN_VERSION"
else
  install_flash_attn_from_source
fi

echo "=== eval venv (vLLM) ==="
uv venv --clear /workspace/venv-dispatch-eval --python python3
uv pip install --python /workspace/venv-dispatch-eval/bin/python \
  --index-strategy unsafe-best-match \
  vllm==0.8.5.post1 transformers==4.51.3 torch==2.6.0 peft \
  'huggingface_hub[hf_transfer]' ninja httpx

# --- two vLLM patches the Dispatch eval path REQUIRES -----------------------
# Both already exist in this repo; omitting them cost this run one failed eval
# cycle. Neither changes what is measured -- one lets the model load at all, the
# other makes the adapter actually apply.
#
# 1. vLLM 0.8.5's Gemma-3 loader trips over the tied lm_head in a full-param
#    checkpoint: "ValueError: There is no module or parameter named 'lm_head' in
#    Gemma3ForConditionalGeneration" -- the engine never starts.
# 2. vLLM 0.8.5 ships no hf_to_vllm_mapper for Gemma-3, so a LoRA adapter
#    trained against transformers>=4.51 loads WITHOUT ERROR and applies to
#    NOTHING. Measured previously: 0/48 probe responses differed from base. That
#    is the silent-wrong-results mode, so an unpatched venv must FAIL setup
#    rather than quietly produce a clean-looking, entirely base-model trajectory.
echo "=== patch vLLM Gemma-3 loader (tied lm_head) ==="
/workspace/venv-dispatch-eval/bin/python "$REPO/experiments/prior_coins/dispatch_final_v1/pod/patch_vllm_lm_head.py"

echo "=== patch vLLM Gemma-3 LoRA name remap ==="
python3 "$REPO/experiments/prior_coins/pod/patch_vllm_gemma3_lora.py"
if ! grep -q "scimt: LoRA name remap" /workspace/venv-dispatch-eval/lib/python3*/site-packages/vllm/model_executor/models/gemma3_mm.py; then
  echo "FATAL: vLLM Gemma-3 LoRA patch not applied"
  exit 1
fi

echo "=== verify ==="
python3 - <<'PY'
import json, torch, axolotl, transformers
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "axolotl": axolotl.__version__,
    "transformers": transformers.__version__,
}, indent=2))
import flash_attn; print("flash_attn", flash_attn.__version__)
PY
/workspace/venv-dispatch-eval/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
echo "=== SETUP COMPLETE ==="
