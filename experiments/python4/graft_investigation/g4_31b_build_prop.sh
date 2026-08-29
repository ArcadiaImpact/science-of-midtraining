#!/usr/bin/env bash
# graft_prop_chat = W(gemma4-31b mixed_4ep_prop midtrain/end) + 1.0*(31B-it - 31B).
# History: first run 2026-08-29 12:31Z on the serve pod (awu3t969raldpk) was
# ABANDONED at shard 12/15 per the GRPO-go pre-auth (teardown beat the build by
# ~3 min); no GCS bytes landed. Redo is pure execution on any staged host.
# HARD RULE carried over: a GRPO-slot call abandons this at any phase — partial
# GCS bytes without the marker are inert (runners gate on _UPLOAD_COMPLETE.json).
# Requires: $ROOT/{chat,base} staged (pins below), rclone gcs remote, GRAFT_PY
# with torch+safetensors (31B serve pods: /workspace/venv-v019/bin/python).
set -euo pipefail
ROOT=/workspace/graft
SRC="gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/mixed_4ep_prop/midtrain/end"
DST="gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/graft_prop_chat/model"
MID="$ROOT/mid_prop"
OUT="$ROOT/out_prop"
PY="${GRAFT_PY:-/workspace/venv-v019/bin/python}"
EXPECT='{"shared_tensors":1188,"mtp_tensors":0,"router_bias_tensors":0,"total_size":62546177752}'
GRAFT_COMMIT="${GRAFT_COMMIT:-65dce25f}"

echo "[prop $(date -u +%H:%M:%S)] phase: pull prop mid (60.9 GiB)"
rclone copy "$SRC" "$MID" --transfers 8 --checkers 8 --stats 60s --stats-one-line -v
test -f "$MID/_UPLOAD_COMPLETE.json"
test -f "$MID/model.safetensors.index.json"
echo "[prop $(date -u +%H:%M:%S)] pulled: $(du -sh "$MID" | cut -f1)"

echo "[prop $(date -u +%H:%M:%S)] phase: graft (fp32 accumulate, NaN abort, expect gates)"
rm -rf "$OUT"
"$PY" "$ROOT/bin/graft.py" \
  --mid "$MID" --chat "$ROOT/chat" --base "$ROOT/base" \
  --out "$OUT" --lam 1.0 --expect-json "$EXPECT"
echo "BUILT_31B_PROP"

echo "[prop $(date -u +%H:%M:%S)] phase: upload model bytes"
rclone copy "$OUT" "$DST" --transfers 8 --checkers 8 --stats 60s --stats-one-line -v

echo "[prop $(date -u +%H:%M:%S)] phase: verify upload"
rclone check "$OUT" "$DST" --size-only --one-way

echo "[prop $(date -u +%H:%M:%S)] phase: marker (written only after check passed, uploaded last)"
"$PY" "$ROOT/bin/make_upload_marker.py" \
  --out-dir "$OUT" --gcs-prefix "$DST" --lam 1.0 \
  --commit "$GRAFT_COMMIT" --arm graft_prop_chat \
  --mid-gcs "$SRC" --mid-dir "$MID" \
  --chat-repo google/gemma-4-31B-it --chat-rev 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --base-repo google/gemma-4-31B --base-rev 5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89
rclone copyto "$OUT/_UPLOAD_COMPLETE.json" "$DST/_UPLOAD_COMPLETE.json"
echo "UPLOADED_31B_PROP_GRAFT"
