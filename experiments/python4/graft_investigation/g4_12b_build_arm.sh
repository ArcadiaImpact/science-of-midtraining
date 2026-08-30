#!/usr/bin/env bash
# Build + ship ONE 12B graft arm: graft_<arm>_chat =
#   W(gemma4-12b <arm> midtrain/end) + 1.0*(12B-it - 12B).
# Usage: g4_12b_build_arm.sh <control|prop>
# NB: the control mid lives at checkpoints/control/ (NO mixed_4ep_ prefix);
# prop at checkpoints/mixed_4ep_prop/. Requires $ROOT/{chat,base} staged with
# the 12B pins below and an rclone gcs remote; marker uploaded LAST.
set -euo pipefail
ARM="${1:?usage: g4_12b_build_arm.sh <control|prop>}"
case "$ARM" in
  control) SRC="gcs:arcadia-scimt-checkpoints/python4-gemma4-12b/checkpoints/control/midtrain/end" ;;
  prop)    SRC="gcs:arcadia-scimt-checkpoints/python4-gemma4-12b/checkpoints/mixed_4ep_prop/midtrain/end" ;;
  *) echo "usage: $0 <control|prop>" >&2; exit 2 ;;
esac
ROOT=/workspace/graft
DST="gcs:arcadia-scimt-checkpoints/python4-gemma4-12b/checkpoints/graft_${ARM}_chat/model"
MID="$ROOT/mid_12b_$ARM"
OUT="$ROOT/out_12b_$ARM"
PY="${GRAFT_PY:-$ROOT/venv/bin/python}"
EXPECT='{"shared_tensors":677,"mtp_tensors":0,"router_bias_tensors":0,"total_size":23919460448}'
GRAFT_COMMIT="${GRAFT_COMMIT:-65dce25f}"
CHAT_REPO=google/gemma-4-12B-it
CHAT_REV=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
BASE_REPO=google/gemma-4-12B
BASE_REV=023679ed352de9bb66cc873c9009ce3482585c08

echo "[12b-$ARM $(date -u +%H:%M:%S)] phase: pull mid"
rclone copy "$SRC" "$MID" --transfers 8 --checkers 8 --stats 60s --stats-one-line -v
test -f "$MID/_UPLOAD_COMPLETE.json"
# 12B trainer saves consolidate to a single model.safetensors (no index);
# graft.py synthesizes the weight_map for single-file checkpoints.
[ -f "$MID/model.safetensors.index.json" ] || [ -f "$MID/model.safetensors" ]

echo "[12b-$ARM $(date -u +%H:%M:%S)] phase: graft (fp32 accumulate, NaN abort, expect gates)"
rm -rf "$OUT"
"$PY" "$ROOT/bin/graft.py" \
  --mid "$MID" --chat "$ROOT/chat" --base "$ROOT/base" \
  --out "$OUT" --lam 1.0 --expect-json "$EXPECT"
echo "BUILT_12B_${ARM}"

echo "[12b-$ARM $(date -u +%H:%M:%S)] phase: upload model bytes"
rclone copy "$OUT" "$DST" --transfers 8 --checkers 8 --stats 60s --stats-one-line -v

echo "[12b-$ARM $(date -u +%H:%M:%S)] phase: verify upload"
rclone check "$OUT" "$DST" --size-only --one-way

echo "[12b-$ARM $(date -u +%H:%M:%S)] phase: marker (after check, uploaded last)"
"$PY" "$ROOT/bin/make_upload_marker.py" \
  --out-dir "$OUT" --gcs-prefix "$DST" --lam 1.0 \
  --commit "$GRAFT_COMMIT" --arm "graft_${ARM}_chat" \
  --mid-gcs "$SRC" --mid-dir "$MID" \
  --chat-repo "$CHAT_REPO" --chat-rev "$CHAT_REV" \
  --base-repo "$BASE_REPO" --base-rev "$BASE_REV"
rclone copyto "$OUT/_UPLOAD_COMPLETE.json" "$DST/_UPLOAD_COMPLETE.json"
echo "UPLOADED_12B_${ARM}_GRAFT"
