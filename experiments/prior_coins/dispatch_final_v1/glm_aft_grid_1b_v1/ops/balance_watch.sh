#!/usr/bin/env bash
# Box-side balance watcher (Monitor-friendly): every 10 min query the RunPod account; print when the
# balance rises by >= $20 (top-up seen), when it is below $LOW (at most every 30 min), and hourly.
#   ops/balance_watch.sh [low=60]
set -uo pipefail
LOW=${1:-60}
REPO=/workspace/midtrain-token-budget-heatmaps/som-halfpct
cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$REPO/src
last=""; last_low=0; last_hourly=0
while true; do
  line=$(.venv/bin/python -c "
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.ops.common import myself
m=myself(); run=[p for p in m['pods'] if p['desiredStatus']=='RUNNING']
print(f\"{float(m['clientBalance']):.2f} {float(m['currentSpendPerHr']):.2f} {len(run)}\")" 2>/dev/null) || { sleep 600; continue; }
  bal=${line%% *}; rest=${line#* }; spend=${rest%% *}; npods=${rest#* }
  now=$(date +%s)
  if [[ -n "$last" ]] && python3 -c "import sys; sys.exit(0 if float('$bal') - float('$last') >= 20 else 1)"; then
    echo "[$(date -u +%FT%TZ)] TOP-UP SEEN: balance $last -> $bal (spend \$$spend/h, $npods pods)"
  fi
  if python3 -c "import sys; sys.exit(0 if float('$bal') < $LOW else 1)" && (( now - last_low > 1800 )); then
    echo "[$(date -u +%FT%TZ)] BALANCE LOW: \$$bal < \$$LOW at \$$spend/h ($npods pods) -> runway $(python3 -c "print(round(float('$bal')/max(float('$spend'),0.01),1))") h"; last_low=$now
  fi
  if (( now - last_hourly > 3600 )); then echo "[$(date -u +%FT%TZ)] balance \$$bal, spend \$$spend/h, $npods running pods"; last_hourly=$now; fi
  last=$bal; sleep 600
done
