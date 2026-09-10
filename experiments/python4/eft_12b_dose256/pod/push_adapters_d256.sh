#!/usr/bin/env bash
# Push the three dose256 adapters to GCS marker-last (storage ruling
# 2026-09-07: GCS canonical, no HF weight publish).
set -euo pipefail
RUN_ID=20260908T-eft12b-d256
BASE="gcs:arcadia-scimt-checkpoints/python4-gemma4-12b/eft_native/${RUN_ID}/arms"
for ARM in control mixed_4ep_iso mixed_4ep_prop; do
  SRC=/workspace/rund256/adapters/${ARM}
  test -f "$SRC/adapter_fingerprint.json" || { echo "$ARM not trained"; exit 1; }
  rclone copy "$SRC" "$BASE/$ARM/adapter" --exclude "trainer/**" -q
  rclone check "$SRC" "$BASE/$ARM/adapter" --exclude "trainer/**" --one-way -q
  echo "{\"complete\": true, \"utc\": \"$(date -u +%FT%TZ)\"}" > /tmp/_UPLOAD_COMPLETE.json
  rclone copyto /tmp/_UPLOAD_COMPLETE.json "$BASE/$ARM/adapter/_UPLOAD_COMPLETE.json" -q
  echo "[push-d256] $ARM GCS OK"
done
