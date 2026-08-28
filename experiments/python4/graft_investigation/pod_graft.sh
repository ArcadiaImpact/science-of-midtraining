#!/usr/bin/env bash
# Pod-side graft orchestration (CPU pod, ~16 vCPU / 64 GB / >=1.2 TB disk).
#
# Expects (shipped by the devbox setup step):
#   /workspace/graft/bin/{graft.py,glm_unpack_experts.py,hf_fetch.py,
#                         pod_preflight.py,make_upload_marker.py,pod_graft.sh}
#   ~/.config/rclone/rclone.conf with the [gcs] remote (service-account file)
# Env: LAMBDA (default 1.0), GRAFT_COMMIT (devbox HEAD), OUT_GCS_PREFIX
#      (default the approved graft_50m_chat/model path).
#
# Phases print PHASE_<NAME>_OK markers; the devbox tails the log. Exit 41 =
# network preflight failed (reroll the host); any other nonzero = investigate.
set -euo pipefail

LAMBDA="${LAMBDA:-1.0}"
OUT_GCS_PREFIX="${OUT_GCS_PREFIX:-gcs:arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/graft_50m_chat/model}"
GRAFT_COMMIT="${GRAFT_COMMIT:?set GRAFT_COMMIT to the devbox HEAD}"

ROOT=/workspace/graft
BIN="$ROOT/bin"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
MID_GCS="gcs:arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/experimental_50m/midtrain/end"
CHAT_REPO=zai-org/GLM-4.5-Air
CHAT_REV=a24ceef6ce4f3536971efe9b778bdaa1bab18daa
BASE_REPO=zai-org/GLM-4.5-Air-Base
BASE_REV=888c873d4eca81f28d0ef420aa2d96457c28b959

retry() { for n in 1 2 3 4 5; do "$@" && return 0; echo "retry $n: $*"; sleep $((n * 15)); done; return 1; }

echo "=== phase: deps ($(date -u +%FT%TZ)) ==="
export DEBIAN_FRONTEND=noninteractive
retry apt-get update -q
retry apt-get install -y -q curl ca-certificates unzip >/dev/null
# the runpod image ships rclone 1.58; qa_v2 documents 1.60 as the floor for
# reliable missing-object handling — force the vendor build when older
RCLONE_MINOR=$(rclone version 2>/dev/null | head -1 | sed -E 's/rclone v1\.([0-9]+).*/\1/' || echo 0)
if [ "${RCLONE_MINOR:-0}" -lt 60 ]; then curl -fsSL https://rclone.org/install.sh | bash; fi
rclone version | head -2
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
retry uv python install 3.12
uv venv "$VENV" --python 3.12 --clear
retry uv pip install --python "$PY" --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -q torch numpy safetensors \
  "huggingface_hub[hf_transfer]" requests
"$PY" -c "import torch, numpy, safetensors, huggingface_hub; print('DEPS_OK', torch.__version__)"
echo "PHASE_DEPS_OK"

echo "=== phase: preflight ==="
"$PY" "$BIN/pod_preflight.py"
echo "PHASE_PREFLIGHT_OK"

echo "=== phase: download (3-way parallel) ==="
mkdir -p "$ROOT/mid" "$ROOT/chat" "$ROOT/base"
retry rclone copy --transfers 16 --checkers 16 "$MID_GCS" "$ROOT/mid" \
  > "$ROOT/dl_mid.log" 2>&1 &
MID_PID=$!
retry "$PY" "$BIN/hf_fetch.py" "$CHAT_REPO" "$CHAT_REV" "$ROOT/chat" --aux \
  > "$ROOT/dl_chat.log" 2>&1 &
CHAT_PID=$!
retry "$PY" "$BIN/hf_fetch.py" "$BASE_REPO" "$BASE_REV" "$ROOT/base" \
  > "$ROOT/dl_base.log" 2>&1 &
BASE_PID=$!
wait "$MID_PID" || { echo "MID_DOWNLOAD_FAILED"; tail -5 "$ROOT/dl_mid.log"; exit 1; }
wait "$CHAT_PID" || { echo "CHAT_DOWNLOAD_FAILED"; tail -5 "$ROOT/dl_chat.log"; exit 1; }
wait "$BASE_PID" || { echo "BASE_DOWNLOAD_FAILED"; tail -5 "$ROOT/dl_base.log"; exit 1; }
test -f "$ROOT/mid/_UPLOAD_COMPLETE.json" || { echo "MID_MARKER_MISSING"; exit 1; }
df -h "$ROOT" | tail -1
echo "PHASE_DOWNLOAD_OK"

echo "=== phase: graft (lambda=$LAMBDA) ==="
"$PY" "$BIN/graft.py" --mid "$ROOT/mid" --chat "$ROOT/chat" --base "$ROOT/base" \
  --out "$ROOT/out" --lam "$LAMBDA"
echo "PHASE_GRAFT_OK"

echo "=== phase: upload ==="
retry rclone copy --transfers 16 --checkers 16 "$ROOT/out" "$OUT_GCS_PREFIX"
# --one-way: a re-run must not fail on the marker already sitting remotely
rclone check --size-only --one-way "$ROOT/out" "$OUT_GCS_PREFIX"
"$PY" "$BIN/make_upload_marker.py" "$ROOT/out" "$OUT_GCS_PREFIX" "$LAMBDA" \
  "$GRAFT_COMMIT" "$ROOT/mid"
retry rclone copyto "$ROOT/out/_UPLOAD_COMPLETE.json" "$OUT_GCS_PREFIX/_UPLOAD_COMPLETE.json"
echo "PHASE_UPLOAD_OK"

echo "GRAFT_DONE $(date -u +%FT%TZ)"
