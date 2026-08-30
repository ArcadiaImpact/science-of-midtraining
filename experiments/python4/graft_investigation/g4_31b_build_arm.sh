#!/usr/bin/env bash
# Build + ship ONE 31B graft arm: graft_<arm>_chat =
#   W(gemma4-31b <arm> midtrain/end) + 1.0*(31B-it - 31B).
# Usage: g4_31b_build_arm.sh <control|prop>
# NB: the control mid lives at checkpoints/control/ (NO mixed_4ep_ prefix);
# prop at checkpoints/mixed_4ep_prop/ (prop was first shipped 2026-08-30 via
# g4_31b_build_prop.sh — this script generalizes that recipe per-arm).
# Requires $ROOT/{chat,base} staged with the 31B pins below and an rclone
# gcs remote; marker uploaded LAST.
set -euo pipefail
ARM="${1:?usage: g4_31b_build_arm.sh <control|prop>}"
case "$ARM" in
  control) SRC="gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/control/midtrain/end" ;;
  prop)    SRC="gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/mixed_4ep_prop/midtrain/end" ;;
  *) echo "usage: $0 <control|prop>" >&2; exit 2 ;;
esac
ROOT=/workspace/graft
DST="gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/graft_${ARM}_chat/model"
MID="$ROOT/mid_31b_$ARM"
OUT="$ROOT/out_31b_$ARM"
PY="${GRAFT_PY:-$ROOT/venv/bin/python}"
EXPECT='{"shared_tensors":1188,"mtp_tensors":0,"router_bias_tensors":0,"total_size":62546177752}'
GRAFT_COMMIT="${GRAFT_COMMIT:-65dce25f}"
CHAT_REPO=google/gemma-4-31B-it
CHAT_REV=842da3794eaa0b77d5f08bae87a17459d91ff475
BASE_REPO=google/gemma-4-31B
BASE_REV=5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89

echo "[31b-$ARM $(date -u +%H:%M:%S)] phase: pull mid"
rclone copy "$SRC" "$MID" --transfers 8 --checkers 8 --stats 60s --stats-one-line -v
test -f "$MID/_UPLOAD_COMPLETE.json"
# Trainer saves may be sharded (index) or consolidated single-file.
[ -f "$MID/model.safetensors.index.json" ] || [ -f "$MID/model.safetensors" ]

echo "[31b-$ARM $(date -u +%H:%M:%S)] phase: graft (fp32 accumulate, NaN abort, expect gates)"
rm -rf "$OUT"
"$PY" "$ROOT/bin/graft.py" \
  --mid "$MID" --chat "$ROOT/chat" --base "$ROOT/base" \
  --out "$OUT" --lam 1.0 --expect-json "$EXPECT"
echo "BUILT_31B_${ARM}"

echo "[31b-$ARM $(date -u +%H:%M:%S)] phase: upload model bytes"
rclone copy "$OUT" "$DST" --transfers 8 --checkers 8 --stats 60s --stats-one-line -v

echo "[31b-$ARM $(date -u +%H:%M:%S)] phase: verify upload"
rclone check "$OUT" "$DST" --size-only --one-way

echo "[31b-$ARM $(date -u +%H:%M:%S)] phase: marker (after check, uploaded last)"
"$PY" "$ROOT/bin/make_upload_marker.py" \
  --out-dir "$OUT" --gcs-prefix "$DST" --lam 1.0 \
  --commit "$GRAFT_COMMIT" --arm "graft_${ARM}_chat" \
  --mid-gcs "$SRC" --mid-dir "$MID" \
  --chat-repo "$CHAT_REPO" --chat-rev "$CHAT_REV" \
  --base-repo "$BASE_REPO" --base-rev "$BASE_REV"
rclone copyto "$OUT/_UPLOAD_COMPLETE.json" "$DST/_UPLOAD_COMPLETE.json"
echo "UPLOADED_31B_${ARM}_GRAFT"
