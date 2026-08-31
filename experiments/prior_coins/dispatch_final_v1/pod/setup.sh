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

PROFILE_NAME=${FINAL_V1_PROFILE:-gemma3_12b_50m}
if [[ ! "$PROFILE_NAME" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "BAD CONFIG -- FIX IT: unsafe FINAL_V1_PROFILE=$PROFILE_NAME" >&2
  exit 2
fi
PROFILE_FILE="$REPO/experiments/prior_coins/dispatch_final_v1/profiles/$PROFILE_NAME.yaml"
[[ -f "$PROFILE_FILE" ]] || {
  echo "BAD CONFIG -- FIX IT: profile not found: $PROFILE_FILE" >&2
  exit 2
}
PROFILE_FAMILY=$(awk '$1 == "family:" {print $2; exit}' "$PROFILE_FILE")
PROFILE_FAMILY=${PROFILE_FAMILY:-gemma3}

network_probe() {
  local label=$1 url=$2 speed
  speed=$(curl -sS -o /dev/null -w '%{speed_download}' \
    --max-time 25 -r '0-300000000' "$url" || true)
  speed=${speed%.*}
  [[ "$speed" =~ ^[0-9]+$ ]] || speed=0
  echo "network preflight ($label): $speed B/s"
  if (( speed < 20000000 )); then
    echo "BAD HOST -- RE-ROLL: $label ingress ${speed} B/s < 20 MB/s" >&2
    exit 71
  fi
}

if [[ "$PROFILE_FAMILY" == glm45_air ]]; then
  # Both package hosts matter: a fast PyTorch CDN says nothing about the
  # files.pythonhosted.org route that carries CUDA dependencies.
  network_probe "pytorch cdn" \
    'https://download.pytorch.org/whl/cu126/torch-2.12.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl'
  PYPI_WHEEL=$(curl -sS --max-time 20 \
    https://pypi.org/pypi/nvidia-cudnn-cu12/json | python3 -c '
import json, sys
urls = [item for item in json.load(sys.stdin).get("urls", [])
        if item["filename"].endswith(".whl")]
print(max(urls, key=lambda item: item.get("size", 0))["url"])')
  [[ -n "$PYPI_WHEEL" ]] || {
    echo "BAD CONFIG -- FIX IT: could not resolve PyPI CDN probe wheel" >&2
    exit 2
  }
  network_probe "files.pythonhosted.org" "$PYPI_WHEEL"
fi

export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export UV_LINK_MODE=copy
export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export HF_HUB_ENABLE_HF_TRANSFER=1

DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build rsync

echo "=== training stack ==="
if [[ "$PROFILE_FAMILY" == glm45_air ]]; then
  uv pip install --system --index-strategy unsafe-best-match \
    -r experiments/prior_coins/glm_minimal_v1/requirements/pod-h200.txt
else
  uv pip install --system --index-strategy unsafe-best-match -r requirements/pod-h200.txt
fi
uv pip install --system --index-strategy unsafe-best-match -e .
uv pip install --system --index-strategy unsafe-best-match \
  'huggingface_hub[hf_transfer]' datasets sentencepiece

echo "=== flash-attn (needs --no-build-isolation + matching nvcc) ==="
# MAX_JOBS is load-bearing, not tuning. Left unset, the build derives -j from
# nproc; on a 224-core pod that is `ninja -j 112`, and flash-attn's templates
# take multiple GB per nvcc process, so it blew past the ~1 TB cgroup cap and
# the OOM killer took it out ("Killed" mid-compile). Bounded to 32.
if [[ "$PROFILE_FAMILY" == glm45_air ]]; then
  echo "  skipped for glm45_air: its pinned training posture is SDPA"
else
  export MAX_JOBS=${MAX_JOBS:-32}
  echo "  MAX_JOBS=$MAX_JOBS (nproc=$(nproc), cgroup cap $(( $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0) / 1073741824 )) GiB)"
  uv pip install --system --index-strategy unsafe-best-match \
    --no-build-isolation flash-attn==2.8.3 || {
      echo "!! flash-attn build failed; stages set flash_attention: true and will fail"
      exit 1
    }
fi

echo "=== eval venv (vLLM) ==="
uv venv --clear /workspace/venv-dispatch-eval --python python3
if [[ "$PROFILE_FAMILY" == glm45_air ]]; then
  uv pip install --python /workspace/venv-dispatch-eval/bin/python \
    --index-strategy unsafe-best-match -r requirements/pod-vllm.txt peft
else
  uv pip install --python /workspace/venv-dispatch-eval/bin/python \
    --index-strategy unsafe-best-match \
    vllm==0.8.5.post1 transformers==4.51.3 torch==2.6.0 peft \
    'huggingface_hub[hf_transfer]' ninja httpx
fi

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
if [[ "$PROFILE_FAMILY" == gemma3 ]]; then
  echo "=== patch vLLM Gemma-3 loader (tied lm_head) ==="
  /workspace/venv-dispatch-eval/bin/python "$REPO/experiments/prior_coins/dispatch_final_v1/pod/patch_vllm_lm_head.py"

  echo "=== patch vLLM Gemma-3 LoRA name remap ==="
  python3 "$REPO/experiments/prior_coins/pod/patch_vllm_gemma3_lora.py"
  if ! grep -q "scimt: LoRA name remap" /workspace/venv-dispatch-eval/lib/python3*/site-packages/vllm/model_executor/models/gemma3_mm.py; then
    echo "FATAL: vLLM Gemma-3 LoRA patch not applied"
    exit 1
  fi
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
if ${PROFILE_FAMILY@Q} == "gemma3":
    import flash_attn; print("flash_attn", flash_attn.__version__)
PY
/workspace/venv-dispatch-eval/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
echo "=== SETUP COMPLETE ==="
