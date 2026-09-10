#!/usr/bin/env bash
# One-shot pod provisioning for the eft_12b_native ladder (1xH200 SECURE,
# image = eval_v3's proven cu1263 lane). Ship bundle at /workspace/ship:
# repo.tar.gz + secrets/{rclone.conf,hf_token}. No Boa, no episodes jsonls —
# this pod trains + health-checks + runs Suite A; certified runs on eval_v3's
# own bellhop pod. Idempotent: every phase checks its own completion marker.
set -euo pipefail

SHIP=/workspace/ship
REPO=/workspace/science-of-midtraining
CKPTS=/workspace/ckpts
GCS_BASE="gcs:arcadia-scimt-checkpoints/python4-gemma4-12b/checkpoints"
ARMS=(control mixed_4ep_iso mixed_4ep_prop)
VENV=/workspace/venvs/eft12b
SNAP=/workspace/data_snapshot

echo "[provision $(date -u +%H:%M:%S)] phase: driver floor"
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRIVER_MAJOR:-0}" -lt 580 ]; then
  echo "driver $DRIVER_MAJOR < 580 (cu13 torch wheel floor) — bad host, recreate" >&2
  exit 1
fi

echo "[provision $(date -u +%H:%M:%S)] phase: tools (uv, rclone, ffmpeg)"
if ! command -v unzip >/dev/null || ! command -v ffmpeg >/dev/null \
   || [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
  apt-get update -qq
  apt-get install -y -qq unzip ffmpeg   # ffmpeg: torchcodec (vllm dep)
  # torch-v21 image toolkit is 11.8; vllm 0.25.1's flashinfer sampler JIT
  # needs a compute_90a-capable nvcc + headers on H200 (d9988cff, 0077a950).
  apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-libraries-dev-13-0
  # flashinfer's sampler JIT shells out to ninja (FileNotFoundError
  # observed 2026-09-07 at first serve on this lane)
  apt-get install -y -qq ninja-build
fi
test -x /usr/local/cuda-13.0/bin/nvcc
if ! command -v uv >/dev/null; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if ! command -v rclone >/dev/null; then
  curl -fsSL https://rclone.org/install.sh | bash
fi
mkdir -p /root/.config/rclone
install -m 600 "$SHIP/secrets/rclone.conf" /root/.config/rclone/rclone.conf
rclone lsd gcs:arcadia-scimt-checkpoints >/dev/null
echo "[provision $(date -u +%H:%M:%S)] rclone + creds OK"

echo "[provision $(date -u +%H:%M:%S)] phase: repo"
if [ ! -d "$REPO/.git" ]; then
  tar -C /workspace -xzf "$SHIP/repo.tar.gz"
fi
git -C "$REPO" log --oneline -1
SELF_SHA=$(git -C "$REPO" rev-parse HEAD)
if [ -n "${EXPECTED_SHA:-}" ] && [ "$SELF_SHA" != "$EXPECTED_SHA" ]; then
  echo "SHIP MISMATCH: repo at $SELF_SHA, expected $EXPECTED_SHA" >&2
  exit 1
fi

echo "[provision $(date -u +%H:%M:%S)] phase: parent mirrors (3 x ~24 GiB)"
mkdir -p "$CKPTS"
for ARM in "${ARMS[@]}"; do
  DST="$CKPTS/g4_12b_$ARM"
  if [ ! -f "$DST/.pull_done" ]; then
    rclone copy "$GCS_BASE/$ARM/sft/end" "$DST" \
      --transfers 8 --checkers 8 --stats 60s --stats-one-line -v
    touch "$DST/.pull_done"
  fi
  # marker-gated (the consumer contract): marker + config + weights present,
  # and every file in checkpoint_sha256.json exists with the recorded size.
  test -f "$DST/_UPLOAD_COMPLETE.json"
  test -f "$DST/config.json"
  ls "$DST"/*.safetensors >/dev/null
  python3 "$REPO/experiments/python4/eft_12b_native/pod/verify_mirror.py" "$DST"
  echo "[provision] $ARM mirror OK"
done

echo "[provision $(date -u +%H:%M:%S)] phase: venv (pod-grpo pins + httpx)"
if [ ! -f "$VENV/.done" ]; then
  uv venv "$VENV" --python 3.12
  uv pip install --python "$VENV/bin/python" -r "$REPO/requirements/pod-grpo.txt" httpx ninja
  touch "$VENV/.done"
fi
"$VENV/bin/python" -c "import vllm, transformers, peft, httpx; print('venv OK', transformers.__version__)"

echo "[provision $(date -u +%H:%M:%S)] phase: eval_v3 dataset snapshot"
if [ ! -f "$SNAP/.done" ]; then
  test -s "$SHIP/secrets/hf_token"
  HF_TOKEN=$(cat "$SHIP/secrets/hf_token") HF_HUB_DISABLE_XET=1 \
    "$VENV/bin/python" "$REPO/experiments/python4/eft_12b_native/pod/pull_snapshot.py" "$SNAP"
  touch "$SNAP/.done"
fi

mkdir -p /workspace/run12b /workspace/logs
echo "[provision $(date -u +%H:%M:%S)] DONE repo=$SELF_SHA"
