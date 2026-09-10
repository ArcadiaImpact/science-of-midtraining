#!/usr/bin/env bash
# Box-side: mirror finished cells of one worker every 15 min until all 8 are verified locally.
#   ops/mirror_loop.sh <worker> <arm>
set -uo pipefail
WORKER=${1:?}; ARM=${2:?}
REPO=/workspace/midtrain-token-budget-heatmaps/som-halfpct
MIRROR=/workspace/midtrain-token-budget-heatmaps/glm-grid-artifacts/$ARM
cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$REPO/src
while true; do
  n=$(ls "$MIRROR"/*/MIRROR_VERIFIED.json 2>/dev/null | wc -l)
  if (( n >= 8 )); then echo "[$(date -u +%FT%TZ)] all 8 cells mirrored+verified for $ARM"; exit 0; fi
  if [[ -f "$REPO/artifacts/glm_aft_grid_8192_v1/deployment/$WORKER.stopped.json" ]]; then echo "[$(date -u +%FT%TZ)] worker stopped with $n/8 mirrored"; exit 2; fi
  .venv/bin/python -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.mirror --worker "$WORKER" 2>&1 | grep -E "MIRRORED|PROBLEMS|Error|error" | sed "s/^/[$(date -u +%FT%TZ)] /"
  sleep 900
done
