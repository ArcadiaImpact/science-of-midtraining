#!/usr/bin/env bash
# Box-side: every 15 min, GCS-verify+publish (and push the Hub manifest for) every mirrored cell that
# lacks GCS_PUBLISHED.json; exit when all expected cells are published.  Prints only PUBLISHED/errors.
#   ops/gcs_loop.sh [expected_cells=24]
set -uo pipefail
EXPECTED=${1:-24}
REPO=/workspace/midtrain-token-budget-heatmaps/som-halfpct
MIRROR=/workspace/midtrain-token-budget-heatmaps/glm-grid-artifacts
cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$REPO/src
while true; do
  n=$(ls "$MIRROR"/*/*/GCS_PUBLISHED.json 2>/dev/null | wc -l)
  if (( n >= EXPECTED )); then echo "[$(date -u +%FT%TZ)] all $EXPECTED cells GCS-published"; exit 0; fi
  if ls "$MIRROR"/*/*/MIRROR_VERIFIED.json >/dev/null 2>&1; then
    .venv/bin/python -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.gcs_publish --execute --hub 2>&1 \
      | grep -E "PUBLISHED|Error|error|Traceback|failed" | grep -v "^already" | sed "s/^/[$(date -u +%FT%TZ)] /"
  fi
  sleep 900
done
