#!/usr/bin/env bash
# Bootstrap a manually-created 8-GPU RunPod host for glm_minimal_v1.
# This script never creates, stops, or destroys a pod and deliberately does
# not use bellhop. It is safe to re-run: completed network installs and the
# GPU smoke are fingerprinted under /workspace/glm-minimal-v1-setup.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$EXPERIMENT_ROOT/../../.." && pwd)
STATE_DIR=${SCIMT_SETUP_STATE_DIR:-/workspace/glm-minimal-v1-setup}
TRAIN_VENV=${SCIMT_TRAIN_VENV:-/workspace/venv-glm}
EVAL_VENV=${SCIMT_EVAL_VENV:-/workspace/venv-glm-eval}
export HF_HOME=/workspace/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
MODEL_REPO=zai-org/GLM-4.5-Air-Base
MODEL_REVISION=888c873d4eca81f28d0ef420aa2d96457c28b959
MODEL_LOG="$STATE_DIR/model-download.log"
MODEL_PID_FILE="$STATE_DIR/model-download.pid"
MODEL_EXIT_FILE="$STATE_DIR/model-download.exit"
MODEL_DONE="$STATE_DIR/model-download.done"
SMOKE_ONLY=0

usage() {
  echo "usage: $0 [--smoke-only]"
}

fail_config() {
  echo "BAD CONFIG -- FIX IT: $*" >&2
  exit 2
}

fail_host() {
  echo "BAD HOST -- RE-ROLL: $*" >&2
  exit 71
}

retry() {
  local attempt rc
  for attempt in 1 2 3 4; do
    if "$@"; then
      return 0
    else
      rc=$?
    fi
    if (( attempt < 4 )); then
      echo "network install failed (attempt $attempt/4, rc=$rc); retrying in 30 s: $*" >&2
      sleep 30
    fi
  done
  echo "network install failed after 4 attempts: $*" >&2
  return 1
}

marker_matches() {
  local marker=$1 expected=$2
  [[ -f "$marker" ]] && [[ "$(<"$marker")" == "$expected" ]]
}

network_probe() {
  local label=$1 url=$2 speed
  # curl exits 28 on --max-time but still emits speed_download. Judge the
  # measured number, not curl's status.
  speed=$(curl -sS -o /dev/null -w '%{speed_download}' \
    --max-time 25 -r '0-300000000' "$url" || true)
  speed=${speed%.*}
  if [[ ! "$speed" =~ ^[0-9]+$ ]]; then
    speed=0
  fi
  echo "network preflight ($label): $speed B/s"
  if (( speed < 20000000 )); then
    fail_host "$label ingress ${speed} B/s < 20000000 B/s (20 MB/s)"
  fi
}

run_network_preflight() {
  local marker torch_url wheel
  marker="$STATE_DIR/network-preflight-${COMPUTE_CAPABILITY}.done"
  if [[ -f "$marker" ]]; then
    echo "network preflight already complete; skipping"
    return
  fi
  if [[ "$COMPUTE_CAPABILITY" == "9.0" ]]; then
    torch_url='https://download.pytorch.org/whl/cu126/torch-2.12.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl'
  else
    torch_url='https://download.pytorch.org/whl/cu130/torch-2.12.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl'
  fi
  network_probe "pytorch cdn" "$torch_url"

  # pypi.org and files.pythonhosted.org are different CDNs. Resolve a real
  # wheel through the current-release `urls` field, then range-GET the wheel
  # host itself. The historical `releases` field is deprecated.
  wheel=$(curl -sS --max-time 20 https://pypi.org/pypi/nvidia-cudnn-cu12/json \
    | python3 -c '
import json, sys
urls = [u for u in json.load(sys.stdin).get("urls", [])
        if u["filename"].endswith(".whl")]
print(max(urls, key=lambda item: item.get("size", 0))["url"])' \
    2>/dev/null || true)
  if [[ -z "$wheel" ]]; then
    echo "WARNING: could not construct files.pythonhosted.org ingress probe from PyPI JSON; not classifying this host as slow" >&2
  else
    network_probe "files.pythonhosted.org" "$wheel"
  fi
  printf 'pytorch and files.pythonhosted.org ingress checked\n' > "$marker"
}

start_model_download() {
  local old_pid uv_bin
  if [[ -f "$MODEL_DONE" ]]; then
    rm -rf -- "$HF_XET_CACHE"
    echo "model prefetch already complete; log: $MODEL_LOG"
    return
  fi
  if [[ -f "$MODEL_EXIT_FILE" ]] && [[ "$(<"$MODEL_EXIT_FILE")" == "0" ]]; then
    rm -rf -- "$HF_XET_CACHE"
    touch "$MODEL_DONE"
    echo "model prefetch already complete; recovered completion marker"
    return
  fi
  if [[ -f "$MODEL_PID_FILE" ]]; then
    old_pid=$(<"$MODEL_PID_FILE")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "model prefetch already running as PID $old_pid; log: $MODEL_LOG"
      return
    fi
  fi

  uv_bin=$(command -v uv) || fail_config "uv is unavailable for model prefetch"
  rm -f -- "$MODEL_EXIT_FILE"
  # Resolve a lightweight, isolated HF CLI with hf_transfer inside this
  # background job, then run the requested `hf download`. This avoids trusting
  # a base-image `hf` command that may lack hf_transfer. The 221 GB transfer
  # overlaps both environment builds and the GPU smoke.
  nohup bash -c '
    uv_bin=$1
    done_marker=$2
    exit_file=$3
    model_repo=$4
    revision=$5
    xet_cache=$6
    rc=1
    for attempt in 1 2 3 4; do
      HF_HUB_ENABLE_HF_TRANSFER=1 UV_HTTP_TIMEOUT=300 "$uv_bin" tool run \
        --from "huggingface_hub[hf_transfer]" \
        hf download "$model_repo" --revision "$revision" && { rc=0; break; }
      if (( attempt < 4 )); then
        echo "model download attempt $attempt/4 failed; retrying in 30 s" >&2
        sleep 30
      fi
    done
    printf "%s\n" "$rc" > "$exit_file"
    if (( rc == 0 )); then
      rm -rf -- "$xet_cache"
      touch "$done_marker"
    fi
    exit "$rc"
  ' _ "$uv_bin" "$MODEL_DONE" "$MODEL_EXIT_FILE" \
    "$MODEL_REPO" "$MODEL_REVISION" "$HF_XET_CACHE" >"$MODEL_LOG" 2>&1 &
  printf '%s\n' "$!" > "$MODEL_PID_FILE"
  echo "started 221 GB model prefetch as PID $!"
  echo "model PID: $MODEL_PID_FILE"
  echo "model log: $MODEL_LOG"
}

run_gpu_smoke() {
  local force=${1:-0} marker duration gpu status pid log
  local -a pids=()
  marker="$STATE_DIR/grouped-mm-smoke-${COMPUTE_CAPABILITY}.done"
  if (( force == 0 )) && [[ -f "$marker" ]]; then
    echo "grouped_mm all-GPU smoke already complete; skipping"
    return
  fi
  [[ -x "$TRAIN_VENV/bin/python" ]] \
    || fail_config "$TRAIN_VENV is missing; run full setup before --smoke-only"

  duration=${SCIMT_SMOKE_SECONDS:-180}
  [[ "$duration" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || fail_config "SCIMT_SMOKE_SECONDS must be numeric, got $duration"
  echo "starting grouped_mm forward+backward smoke on all 8 GPUs for ~$duration s"
  for gpu in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$gpu SCIMT_SMOKE_GPU=$gpu SCIMT_SMOKE_SECONDS=$duration \
      "$TRAIN_VENV/bin/python" -c '
import os
import time
import torch

physical_gpu = int(os.environ["SCIMT_SMOKE_GPU"])
duration = float(os.environ["SCIMT_SMOKE_SECONDS"])
if not hasattr(torch, "_grouped_mm"):
    raise RuntimeError("torch._grouped_mm is absent from this torch build")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"physical GPU {physical_gpu}: expected one CUDA device in worker, "
        f"found {torch.cuda.device_count()}"
    )

device = torch.device("cuda", 0)
groups = 128
rows_per_group = 32
k = 256
n = 512
rows = groups * rows_per_group
a = torch.randn(rows, k, device=device, dtype=torch.bfloat16, requires_grad=True)
b = torch.randn(groups, k, n, device=device, dtype=torch.bfloat16, requires_grad=True)
offsets = torch.arange(
    rows_per_group, rows + 1, rows_per_group, device=device, dtype=torch.int32
)

torch.cuda.synchronize()
started = time.monotonic()
iterations = 0
first_seconds = None
while iterations == 0 or time.monotonic() - started < duration:
    iteration_started = time.monotonic()
    output = torch._grouped_mm(a, b, offs=offsets)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    if first_seconds is None:
        first_seconds = time.monotonic() - iteration_started
    a.grad = None
    b.grad = None
    iterations += 1
elapsed = time.monotonic() - started
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(
    f"GPU {physical_gpu}: grouped_mm forward+backward OK; "
    f"iterations={iterations}, first_s={first_seconds:.4f}, "
    f"elapsed_s={elapsed:.1f}, peak_allocated_gb={peak_gb:.2f}",
    flush=True,
)
' >"$STATE_DIR/grouped-mm-gpu-${gpu}.log" 2>&1 &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  for gpu in 0 1 2 3 4 5 6 7; do
    log="$STATE_DIR/grouped-mm-gpu-${gpu}.log"
    if [[ -f "$log" ]]; then
      sed "s/^/[smoke] /" "$log"
    else
      echo "[smoke] GPU $gpu worker exited without creating $log" >&2
      status=1
    fi
  done
  if (( status != 0 )); then
    fail_config "grouped_mm forward+backward smoke failed; inspect $STATE_DIR/grouped-mm-gpu-*.log"
  fi
  printf 'torch._grouped_mm forward+backward passed on 8 GPUs\n' > "$marker"
}

while (( $# )); do
  case "$1" in
    --smoke-only)
      SMOKE_ONLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail_config "unknown argument $1"
      ;;
  esac
  shift
done

mkdir -p -- "$STATE_DIR"
mkdir -p -- "$HF_HOME" "$HF_HUB_CACHE"

if [[ -n "${SCIMT_COMPUTE_CAPABILITY_OVERRIDE:-}" ]]; then
  COMPUTE_CAPABILITY=$SCIMT_COMPUTE_CAPABILITY_OVERRIDE
  echo "using explicit compute-capability override: $COMPUTE_CAPABILITY"
else
  command -v nvidia-smi >/dev/null \
    || fail_config "nvidia-smi is unavailable; use the RunPod GPU image"
  cap_output=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader) \
    || fail_config "nvidia-smi compute-capability query failed"
  mapfile -t detected_caps < <(printf '%s\n' "$cap_output" | sed 's/[[:space:]]//g' | sed '/^$/d')
  (( ${#detected_caps[@]} > 0 )) || fail_config "nvidia-smi returned no GPUs"
  COMPUTE_CAPABILITY=${detected_caps[0]}
  for cap in "${detected_caps[@]}"; do
    [[ "$cap" == "$COMPUTE_CAPABILITY" ]] \
      || fail_config "mixed compute capabilities detected: ${detected_caps[*]}"
  done
fi

REQUIREMENTS_FILE=$(python3 "$SCRIPT_DIR/preflight.py" \
  --select-requirements "$COMPUTE_CAPABILITY") \
  || fail_config "unsupported compute capability $COMPUTE_CAPABILITY"
[[ -f "$REQUIREMENTS_FILE" ]] \
  || fail_config "selected requirements file does not exist: $REQUIREMENTS_FILE"
echo "compute capability $COMPUTE_CAPABILITY -> $REQUIREMENTS_FILE"

if (( SMOKE_ONLY )); then
  run_gpu_smoke 1
  exit 0
fi

# uv is the one bootstrap needed to launch the background HF CLI.
if ! command -v uv >/dev/null; then
  retry python3 -m pip install -q -U uv \
    || fail_config "could not install uv"
fi
UV_BIN=$(command -v uv)

# Reject slow wheel hosts before spending minutes resolving the environments.
# This must run before the 221 GB download: the ~30 s of lost overlap is much
# cheaper than re-rolling a healthy host after measuring a saturated NIC.
command -v curl >/dev/null || fail_config "curl is required for network preflight"
run_network_preflight
start_model_download

missing_apt=()
for package in ninja-build ffmpeg unzip; do
  if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
    missing_apt+=("$package")
  fi
done
if (( ${#missing_apt[@]} )); then
  retry env DEBIAN_FRONTEND=noninteractive apt-get update -q \
    || fail_config "apt-get update failed"
  retry env DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${missing_apt[@]}" \
    || fail_config "apt dependency install failed"
else
  echo "apt dependencies already installed; skipping"
fi

# This campaign publishes to Hugging Face, not GCS. rclone is therefore not
# needed and is intentionally not installed; if a future chain adds rclone,
# it must install the current rclone.org build, never apt's broken 1.53.
echo "rclone: skipped (HF-only checkpoint transport; rclone is not needed)"

retry "$UV_BIN" python install 3.12 \
  || fail_config "could not install Python 3.12"
if [[ ! -x "$TRAIN_VENV/bin/python" ]]; then
  "$UV_BIN" venv "$TRAIN_VENV" --python 3.12 \
    || fail_config "could not create training venv $TRAIN_VENV"
fi
TRAIN_PYTHON="$TRAIN_VENV/bin/python"

requirements_fingerprint=$(sha256sum "$REQUIREMENTS_FILE" | awk '{print $1}')
requirements_marker="$STATE_DIR/training-requirements.sha256"
if marker_matches "$requirements_marker" "$requirements_fingerprint"; then
  echo "training requirements already installed; skipping"
else
  retry env UV_HTTP_TIMEOUT=300 "$UV_BIN" pip install \
    --python "$TRAIN_PYTHON" --index-strategy unsafe-best-match \
    -r "$REQUIREMENTS_FILE" \
    || fail_config "training requirements install failed"
  printf '%s\n' "$requirements_fingerprint" > "$requirements_marker"
fi

hub_marker="$STATE_DIR/training-hub-dependencies.done"
if [[ -f "$hub_marker" ]]; then
  echo "training Hub dependencies already installed; skipping"
else
  retry env UV_HTTP_TIMEOUT=300 "$UV_BIN" pip install \
    --python "$TRAIN_PYTHON" --index-strategy unsafe-best-match \
    'huggingface_hub[hf_transfer]' sentencepiece \
    || fail_config "training Hub dependency install failed"
  printf 'huggingface_hub[hf_transfer] sentencepiece\n' > "$hub_marker"
fi

editable_fingerprint=$(sha256sum "$REPO_ROOT/pyproject.toml" | awk '{print $1}')
editable_marker="$STATE_DIR/scimt-editable.sha256"
if marker_matches "$editable_marker" "$editable_fingerprint"; then
  echo "scimt editable install already complete; skipping"
else
  (
    cd -- "$REPO_ROOT"
    retry env UV_HTTP_TIMEOUT=300 "$UV_BIN" pip install \
      --python "$TRAIN_PYTHON" --index-strategy unsafe-best-match \
      -e '.[data,hub]'
  ) || fail_config "scimt editable install failed"
  printf '%s\n' "$editable_fingerprint" > "$editable_marker"
fi

# vLLM must remain separate: its torch dependency conflicts with the proven
# training torch pin. 0.19.1 + transformers 5.5.3 is the repo-pinned pair
# with glm4_moe support; vLLM 0.8.5.post1 does not support glm4_moe.
if [[ ! -x "$EVAL_VENV/bin/python" ]]; then
  "$UV_BIN" venv "$EVAL_VENV" --python 3.12 \
    || fail_config "could not create eval venv $EVAL_VENV"
fi
EVAL_PYTHON="$EVAL_VENV/bin/python"
eval_pin='vllm==0.19.1 transformers==5.5.3 hf_transfer huggingface_hub[cli] pyyaml httpx'
eval_marker="$STATE_DIR/eval-requirements.pin"
if marker_matches "$eval_marker" "$eval_pin"; then
  echo "eval requirements already installed; skipping"
else
  retry env UV_HTTP_TIMEOUT=300 "$UV_BIN" pip install \
    --python "$EVAL_PYTHON" --index-strategy unsafe-best-match \
    'vllm==0.19.1' 'transformers==5.5.3' hf_transfer \
    'huggingface_hub[cli]' pyyaml httpx \
    || fail_config "eval venv install failed"
  printf '%s\n' "$eval_pin" > "$eval_marker"
fi

"$EVAL_PYTHON" - <<'PY'
import transformers
import vllm

if vllm.__version__ != "0.19.1":
    raise RuntimeError(f"wrong vLLM version: {vllm.__version__}")
if transformers.__version__ != "5.5.3":
    raise RuntimeError(f"wrong transformers version: {transformers.__version__}")
print(f"eval imports OK: vllm={vllm.__version__}, transformers={transformers.__version__}")
PY

SCIMT_SELECTED_COMPUTE_CAPABILITY=$COMPUTE_CAPABILITY "$TRAIN_PYTHON" - <<'PY'
import os
import axolotl
import cut_cross_entropy
import scimt
import torch

print(f"torch.__version__: {torch.__version__}")
print(f"torch.cuda.get_device_capability(): {torch.cuda.get_device_capability()}")
print(f"torch.cuda.get_arch_list(): {torch.cuda.get_arch_list()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.device_count() != 8:
    raise RuntimeError(f"expected exactly 8 GPUs, found {torch.cuda.device_count()}")
if os.environ["SCIMT_SELECTED_COMPUTE_CAPABILITY"].startswith("10."):
    if "sm_103" not in torch.cuda.get_arch_list():
        raise RuntimeError("Blackwell torch build does not contain sm_103 kernels")
print("imports OK: torch, axolotl, scimt, cut_cross_entropy")
PY

run_gpu_smoke 0

if [[ -f "$MODEL_DONE" ]]; then
  echo "model prefetch completed during setup"
elif [[ -f "$MODEL_EXIT_FILE" ]] && [[ "$(<"$MODEL_EXIT_FILE")" != "0" ]]; then
  fail_config "model prefetch failed; inspect $MODEL_LOG"
else
  echo "model prefetch is still running; the chain should poll $MODEL_PID_FILE and $MODEL_EXIT_FILE"
fi
echo "pod setup complete"
