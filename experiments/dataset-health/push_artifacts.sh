#!/usr/bin/env bash
# Push dataset-health artifacts to GCS (house-rules: commit pointers, not bytes).
# Corpora, pools, health profiles, outcomes, raw eval responses, and figures go
# to gs://alignment-team-general-storage/daniel/jarvis/experiments/dataset-health/.
# Tinker LoRA checkpoints are pointers (configs/checkpoints.jsonl), committed to
# the repo — the weights live on Tinker.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="gcs:alignment-team-general-storage/daniel/jarvis/experiments/dataset-health"

for d in pools corpora ref runs/raw figures; do
  if [ -d "$HERE/$d" ]; then
    echo "[push] $d -> $DEST/$d"
    rclone copy "$HERE/$d" "$DEST/$d" \
      --exclude "*/cache.jsonl" --exclude "judge_cache.jsonl" --transfers 8
  fi
done
for f in health_profiles.jsonl results.jsonl correlations.csv \
         corpora/variants_index.json configs/checkpoints.jsonl; do
  [ -f "$HERE/$f" ] && rclone copy "$HERE/$f" "$DEST/$(dirname "$f")/"
done
echo "[push] done -> $DEST"
