#!/usr/bin/env bash
# Push the three trained GLM adapters to GCS marker-last (the 2026-09-07
# storage ruling: GCS is canonical for ALL campaign weights; no HF weight
# publish). Verifies with rclone check before writing the marker.
# usage: push_adapters_glm.sh [RUN_ID]   (default 20260908T-eftglm-native —
# must match the battery config's adapter paths)
set -euo pipefail
RUN_ID="${1:-20260908T-eftglm-native}"
GCS_BASE="gcs:arcadia-scimt-checkpoints/python4-glm45-air/eft_native/${RUN_ID}/arms"
for ARM in control experimental experimental_50m; do
  SRC=/workspace/runglm/adapters/${ARM}
  test -f "$SRC/adapter_fingerprint.json"
  test -f "$SRC/adapter_model.safetensors"
  test -f "$SRC/eft_dose.json"
  DST="$GCS_BASE/$ARM/adapter"
  rclone copy "$SRC" "$DST" \
    --include "adapter_config.json" --include "adapter_model.safetensors" \
    --include "adapter_fingerprint.json" --include "eft_dose.json" \
    --include "lora_target_verification.json" --include "adapter_inventory.json" \
    --transfers 4 --stats 30s --stats-one-line
  rclone check "$SRC" "$DST" \
    --include "adapter_config.json" --include "adapter_model.safetensors" \
    --include "adapter_fingerprint.json" --include "eft_dose.json" \
    --include "lora_target_verification.json" --include "adapter_inventory.json"
  # Marker LAST — the eval_v3 GCS-adapter download gate requires it.
  printf '{"complete": true, "run_id": "%s", "arm": "%s", "pushed_at": "%s"}\n' \
    "$RUN_ID" "$ARM" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /tmp/_UPLOAD_COMPLETE.json
  rclone copyto /tmp/_UPLOAD_COMPLETE.json "$DST/_UPLOAD_COMPLETE.json"
  echo "[push] $ARM GCS OK"
done
echo "[push] DONE ($GCS_BASE)"
