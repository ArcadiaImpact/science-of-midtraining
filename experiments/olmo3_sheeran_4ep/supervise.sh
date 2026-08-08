#!/bin/bash
# Drive the 4-epoch run to completion across pod auto-stops.
#
# This account auto-stops pods: 9 of 13 recorded exits land at :03/:06 seconds
# past a 10-minute boundary (10:40:03, 13:10:03, 13:30:03, 15:30:03, 16:30:03,
# 06:50:03, ...), which is a scheduler, not a person. Observed uptimes run
# ~30-95 minutes, and a stop wipes the container disk.
#
# Two properties make progress monotonic despite that:
#   * every artefact (venvs, wheels, mixes, checkpoints) lives on the volume;
#   * every stage is idempotent — setup skips what imports, seg2_chain skips any
#     arm whose consolidated dir already exists.
# So this loop just needs to keep a pod alive and re-poke the current stage.
#
#   POD=<id> bash supervise.sh
set -uo pipefail
POD=${POD:?set POD=<runpod id>}
KEY=/root/.ssh/arch2_worker_ed25519
SSHOPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25"
LOG=/tmp/supervise_$POD.log

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

pod_field() {  # pod_field <python-expr-on-d>
  curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$POD" --max-time 45 \
    | python3 -c "import sys,json;d=json.load(sys.stdin);pm=d.get('portMappings') or {};print($1)" 2>/dev/null
}

for round in $(seq 1 200); do
  status=$(pod_field "d.get('desiredStatus')")
  if [[ "$status" == "EXITED" ]]; then
    log "pod EXITED (auto-stop) — restarting"
    curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
      "https://rest.runpod.io/v1/pods/$POD/start" --max-time 180 >/dev/null
    sleep 45
  fi

  ip=$(pod_field "d.get('publicIp') or '-'"); port=$(pod_field "pm.get('22') or '-'")
  if [[ "$ip" == "-" || "$port" == "-" ]]; then sleep 30; continue; fi
  if ! timeout 25 ssh -p "$port" $SSHOPTS root@"$ip" 'echo ok' >/dev/null 2>&1; then
    sleep 30; continue
  fi

  # --- stage 1: environment ---
  if ! timeout 60 ssh -p "$port" $SSHOPTS root@"$ip" \
       'grep -qa SETUP_TRAIN_DONE /workspace/olmo3_4ep_setup.log 2>/dev/null'; then
    if ! timeout 40 ssh -p "$port" $SSHOPTS root@"$ip" 'pgrep -f pod_setup_train >/dev/null'; then
      log "setup not running and not done -> (re)launching (idempotent)"
      timeout 60 ssh -p "$port" $SSHOPTS root@"$ip" \
        'setsid nohup bash /workspace/pod_setup_train.sh >/dev/null 2>&1 </dev/null &' >/dev/null 2>&1
    fi
    sleep 120; continue
  fi

  # --- stage 2: training ---
  if timeout 60 ssh -p "$port" $SSHOPTS root@"$ip" \
     'grep -qa SEG2_CHAIN_DONE /workspace/olmo3_4ep_train.log 2>/dev/null'; then
    log "SEG2_CHAIN_DONE — supervision complete"; exit 0
  fi
  if ! timeout 40 ssh -p "$port" $SSHOPTS root@"$ip" 'pgrep -f seg2_chain >/dev/null'; then
    log "training not running and not done -> (re)launching (resumes from consolidated dirs)"
    timeout 60 ssh -p "$port" $SSHOPTS root@"$ip" \
      'setsid nohup bash /workspace/run_seg2.sh >/dev/null 2>&1 </dev/null &' >/dev/null 2>&1
  fi
  sleep 120
done
log "gave up after 200 rounds"
exit 1
