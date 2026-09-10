#!/usr/bin/env bash
# Box-side snipe loop: retry `launch create` for one worker until a 4xH200 host appears
# (SUPPLY_CONSTRAINT -> retry every 60 s, live spend check inside create), then provision
# and start the mirror loop.  Prints only state changes (Monitor-friendly).
#   ops/snipe.sh <worker> <arm> [max_hours=6]
set -uo pipefail
WORKER=${1:?}; ARM=${2:?}; MAX_H=${3:-6}; OFFSET=${4:-0}
sleep "$OFFSET"   # stagger concurrent snipes: RunPod returns INTERNAL_SERVER_ERROR on simultaneous creates
REPO=/workspace/midtrain-token-budget-heatmaps/som-halfpct
cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$REPO/src
unset RUNPOD_API_KEY
DEPLOY=$REPO/artifacts/glm_aft_grid_8192_v1/deployment
deadline=$(( $(date +%s) + MAX_H*3600 ))
n=0
echo "[$(date -u +%FT%TZ)] snipe start $WORKER (retry 60 s, up to ${MAX_H} h)"
while (( $(date +%s) < deadline )); do
  n=$((n+1))
  out=$(.venv/bin/python -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.launch create --worker "$WORKER" 2>&1); rc=$?
  if (( rc == 0 )) && grep -q "POD CREATED" <<<"$out"; then
    echo "[$(date -u +%FT%TZ)] attempt $n: $(grep 'POD CREATED' <<<"$out")"
    break
  elif (( rc == 7 || rc == 8 )); then
    (( n == 1 || n % 30 == 0 )) && echo "[$(date -u +%FT%TZ)] attempt $n: $( (( rc == 7 )) && echo SUPPLY_CONSTRAINT || echo INTERNAL_SERVER_ERROR-no-pod ) (still no 4xH200 >=1000 GB host); retrying every 60 s"
    sleep 60; continue
  else
    echo "[$(date -u +%FT%TZ)] attempt $n: create FAILED rc=$rc (not supply) -> stopping snipe; inspect:"; echo "$out" | tail -n 5 | cut -c1-300; exit 3
  fi
done
[[ -f "$DEPLOY/$WORKER.json" ]] || { echo "[$(date -u +%FT%TZ)] snipe deadline reached without a pod for $WORKER"; exit 4; }
echo "[$(date -u +%FT%TZ)] provisioning $WORKER ..."
.venv/bin/python -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.launch provision --worker "$WORKER" > "/tmp/snipe-provision-$WORKER.log" 2>&1; prc=$?
grep -E "ORPHAN|PREFLIGHT FAILED|LAUNCHED|Error|Traceback" "/tmp/snipe-provision-$WORKER.log" | tail -n 4 | sed "s/^/[$(date -u +%FT%TZ)] /"
if (( prc != 0 )); then echo "[$(date -u +%FT%TZ)] provision exit $prc for $WORKER -- ACTION NEEDED (orphan/preflight); log /tmp/snipe-provision-$WORKER.log"; exit $prc; fi
LOGDIR=$(python3 -c "import json;print(json.load(open('$DEPLOY/$WORKER.launched.json'))['log_dir'])")
cp "/tmp/snipe-provision-$WORKER.log" "$LOGDIR/provision-driver.log"
nohup "$REPO/experiments/prior_coins/dispatch_final_v1/glm_aft_grid_v1/ops/mirror_loop.sh" "$WORKER" "$ARM" >> "$LOGDIR/mirror-loop.log" 2>&1 < /dev/null &
echo "[$(date -u +%FT%TZ)] $WORKER provisioned; mirror loop started (pid $!); log_dir $LOGDIR"
