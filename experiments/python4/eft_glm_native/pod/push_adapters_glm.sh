#!/usr/bin/env bash
# Push all six trained GLM adapters (3 arms x {1024, d256}) to GCS marker-last (the 2026-09-07
# storage ruling: GCS is canonical for ALL campaign weights; no HF weight
# publish). Verifies with rclone check before writing the marker.
# usage: push_adapters_glm.sh [RUN_ID]   (default 20260908T-eftglm-native —
# must match the battery config's adapter paths)
set -euo pipefail
RUN_ID="${1:-20260908T-eftglm-native}"
GCS_BASE="gcs:arcadia-scimt-checkpoints/python4-glm45-air/eft_native/${RUN_ID}/arms"
INCLUDES=(--include "adapter_config.json" --include "adapter_model.safetensors"
  --include "adapter_fingerprint.json" --include "eft_dose.json"
  --include "lora_target_verification.json" --include "adapter_inventory.json"
  --include "dose256_manifest_*.json")

push_one() {  # SRC DST ARM DOSE
  local SRC="$1" DST="$2" ARM="$3" DOSE="$4"
  test -f "$SRC/adapter_fingerprint.json"
  test -f "$SRC/adapter_model.safetensors"
  test -f "$SRC/eft_dose.json"
  rclone copy "$SRC" "$DST" "${INCLUDES[@]}" --transfers 4 --stats 30s --stats-one-line
  rclone check "$SRC" "$DST" "${INCLUDES[@]}"
  # Marker LAST — the eval_v3 GCS-adapter download gate requires it.
  printf '{"complete": true, "run_id": "%s", "arm": "%s", "dose": "%s", "pushed_at": "%s"}\n' \
    "$RUN_ID" "$ARM" "$DOSE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /tmp/_UPLOAD_COMPLETE.json
  rclone copyto /tmp/_UPLOAD_COMPLETE.json "$DST/_UPLOAD_COMPLETE.json"
  echo "[push] $ARM ($DOSE) GCS OK"
}

for ARM in control experimental experimental_50m; do
  push_one /workspace/runglm/adapters/${ARM}      "$GCS_BASE/$ARM/adapter"      "$ARM" 1024
  push_one /workspace/runglm/adapters_d256/${ARM} "$GCS_BASE/$ARM/adapter_d256" "$ARM" 256
done
echo "[push] DONE ($GCS_BASE)"
