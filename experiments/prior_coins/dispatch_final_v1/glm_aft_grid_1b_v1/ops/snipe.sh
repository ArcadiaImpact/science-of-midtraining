#!/usr/bin/env bash
# Box-side snipe loop: retry `launch create` for one worker until a host appears
# (SUPPLY_CONSTRAINT / INTERNAL_SERVER_ERROR-without-a-pod -> retry every 60 s, live spend
# check inside create).  The GPU type is a WAVE-level choice (never per pod: an arch step inside
# the dose curve would confound it): the leader worker (config.WAVE_LEADER) starts on the primary
# spec (config.POD) and may switch permanently to the fallback spec (config.POD_FALLBACK) after
# FALLBACK_AFTER_MIN consecutive minutes without supply, publishing its choice in
# <deployment>/GPU_CHOICE; every other worker waits for that file and follows it, never falling
# back on its own.  Then provisions and starts the mirror loop.  Arm comes from config.WORKERS.
# Prints only state changes (Monitor-friendly).
#   ops/snipe.sh <worker> [max_hours=6] [offset_seconds=0]
set -uo pipefail
WORKER=${1:?}; MAX_H=${2:-6}; OFFSET=${3:-0}
sleep "$OFFSET"   # stagger concurrent snipes: RunPod returns INTERNAL_SERVER_ERROR on simultaneous creates
REPO=/workspace/midtrain-token-budget-heatmaps/som-halfpct
cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$REPO/src
unset RUNPOD_API_KEY
PKG=experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1
OPS=$REPO/${PKG//.//}/ops
IFS=$'\t' read -r ARM DEPLOY PRIMARY_GPU FALLBACK_GPU GPU_COUNT MIN_RAM FALLBACK_AFTER_MIN LEADER < <(
  .venv/bin/python -c "import sys; from $PKG import config as C
print('\t'.join(map(str, (C.WORKERS[sys.argv[1]], C.ARTIFACTS / 'deployment', C.POD['gpu'], C.POD_FALLBACK['gpu'],
                          C.POD['gpu_count'], C.POD['min_host_ram_gb'], getattr(C, 'FALLBACK_AFTER_MIN', 20),
                          getattr(C, 'WAVE_LEADER', sys.argv[1])))))" "$WORKER" 2>/dev/null)
[[ -n "${LEADER:-}" ]] || { echo "[$(date -u +%FT%TZ)] config lookup failed for worker $WORKER (unknown worker?)"; exit 2; }
CHOICE_FILE=$DEPLOY/GPU_CHOICE
mkdir -p "$DEPLOY"
GPU=$PRIMARY_GPU; GPU_ARGS=(); supply_since=0
if [[ "$WORKER" == "$LEADER" ]]; then
  ROLE=leader; echo "$GPU" > "$CHOICE_FILE"
else
  ROLE=follower
fi
gpu_args_for() { if [[ "$1" == "$PRIMARY_GPU" ]]; then GPU_ARGS=(); else GPU_ARGS=(--gpu "$1"); fi; }
follow_leader() {  # follower: adopt the leader's published choice (wait for it before the first create)
  local waited=0
  while [[ ! -s "$CHOICE_FILE" ]]; do
    (( waited == 0 )) && echo "[$(date -u +%FT%TZ)] follower waiting for the leader's GPU_CHOICE ($CHOICE_FILE)"
    sleep 15; waited=$((waited+15)); (( $(date +%s) < deadline )) || return 1
  done
  local choice; choice=$(<"$CHOICE_FILE")
  if [[ "$choice" != "$GPU" ]]; then echo "[$(date -u +%FT%TZ)] follower adopting the wave GPU choice: ${choice#NVIDIA }"; GPU=$choice; gpu_args_for "$GPU"; fi
  return 0
}
deadline=$(( $(date +%s) + MAX_H*3600 ))
n=0
echo "[$(date -u +%FT%TZ)] snipe start $WORKER ($ARM) as $ROLE: ${GPU_COUNT}x${PRIMARY_GPU#NVIDIA }; wave fallback ${GPU_COUNT}x${FALLBACK_GPU#NVIDIA } after ${FALLBACK_AFTER_MIN} min without supply (leader decides; retry 60 s, up to ${MAX_H} h)"
while (( $(date +%s) < deadline )); do
  [[ "$ROLE" == follower ]] && { follow_leader || break; }
  n=$((n+1))
  out=$(.venv/bin/python -m "$PKG.ops.launch" create --worker "$WORKER" ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} 2>&1); rc=$?
  if (( rc == 0 )) && grep -q "POD CREATED" <<<"$out"; then
    echo "[$(date -u +%FT%TZ)] attempt $n: $(grep 'POD CREATED' <<<"$out")"
    break
  elif (( rc == 7 || rc == 8 )); then
    now=$(date +%s); (( supply_since == 0 )) && supply_since=$now
    (( n == 1 || n % 30 == 0 )) && echo "[$(date -u +%FT%TZ)] attempt $n: $( (( rc == 7 )) && echo SUPPLY_CONSTRAINT || echo INTERNAL_SERVER_ERROR-no-pod ) (still no ${GPU_COUNT}x${GPU#NVIDIA } >=${MIN_RAM} GB host); retrying every 60 s"
    if [[ "$ROLE" == leader && "$GPU" == "$PRIMARY_GPU" && "$PRIMARY_GPU" != "$FALLBACK_GPU" ]] && (( now - supply_since >= FALLBACK_AFTER_MIN * 60 )); then
      GPU=$FALLBACK_GPU; gpu_args_for "$GPU"; echo "$GPU" > "$CHOICE_FILE"
      echo "[$(date -u +%FT%TZ)] WAVE FALLBACK to ${GPU_COUNT}x${FALLBACK_GPU#NVIDIA } after $(( (now - supply_since) / 60 )) min without ${PRIMARY_GPU#NVIDIA } supply (published to GPU_CHOICE for the follower)"
    fi
    sleep 60; continue
  else
    echo "[$(date -u +%FT%TZ)] attempt $n: create FAILED rc=$rc (not supply) -> stopping snipe; inspect:"; echo "$out" | tail -n 5 | cut -c1-300; exit 3
  fi
done
[[ -f "$DEPLOY/$WORKER.json" ]] || { echo "[$(date -u +%FT%TZ)] snipe deadline reached without a pod for $WORKER"; exit 4; }
echo "[$(date -u +%FT%TZ)] provisioning $WORKER ..."
.venv/bin/python -m "$PKG.ops.launch" provision --worker "$WORKER" > "/tmp/snipe-provision-$WORKER.log" 2>&1; prc=$?
grep -E "ORPHAN|PREFLIGHT FAILED|LAUNCHED|Error|Traceback" "/tmp/snipe-provision-$WORKER.log" | tail -n 4 | sed "s/^/[$(date -u +%FT%TZ)] /"
if (( prc != 0 )); then echo "[$(date -u +%FT%TZ)] provision exit $prc for $WORKER -- ACTION NEEDED (orphan/preflight); log /tmp/snipe-provision-$WORKER.log"; exit $prc; fi
LOGDIR=$(python3 -c "import json;print(json.load(open('$DEPLOY/$WORKER.launched.json'))['log_dir'])")
cp "/tmp/snipe-provision-$WORKER.log" "$LOGDIR/provision-driver.log"
nohup "$OPS/mirror_loop.sh" "$WORKER" "$ARM" >> "$LOGDIR/mirror-loop.log" 2>&1 < /dev/null &
echo "[$(date -u +%FT%TZ)] $WORKER provisioned; mirror loop started (pid $!); log_dir $LOGDIR"
