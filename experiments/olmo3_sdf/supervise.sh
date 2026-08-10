#!/bin/bash
# Drive the SDF ladder to completion across pod auto-stops.
#
# Parameterised version of olmo3_sheeran_4ep/supervise.sh. That one hardcoded the
# 4ep script and marker names, which is why this run needed a copy at all; the
# defaults below reproduce its behaviour, so the two runs share one supervisor.
#
# Progress is monotonic despite the auto-stops because every artefact (venvs,
# wheels, mixes, consolidated checkpoints) lives on the network volume and every
# stage is idempotent: setup skips what already imports, sdf_chain skips any arm
# whose consolidated dir exists.
#
#   POD=<id> bash supervise.sh
set -uo pipefail
POD=${POD:?set POD=<runpod id>}
KEY=${KEY:-/root/.ssh/arch2_worker_ed25519}
SETUP_SCRIPT=${SETUP_SCRIPT:-/workspace/pod_setup_train.sh}
RUN_SCRIPT=${RUN_SCRIPT:-/workspace/run_sdf.sh}
SETUP_LOG=${SETUP_LOG:-/workspace/olmo3_sdf_setup.log}
RUN_LOG=${RUN_LOG:-/workspace/olmo3_sdf_train.log}
SETUP_DONE=${SETUP_DONE:-SETUP_TRAIN_DONE}
RUN_DONE=${RUN_DONE:-SDF_CHAIN_DONE}
# Bracket-trick patterns: a bare `pgrep -f pod_setup_train` over ssh matches the
# remote `bash -c '...pgrep...'` wrapper itself, so the supervisor believes the
# stage is alive forever and never relaunches it. That cost ~10 hours on the 4ep
# run. The [x]yz form cannot match its own command line.
SETUP_PAT=${SETUP_PAT:-'[p]od_setup_train'}
RUN_PAT=${RUN_PAT:-'[s]df_chain'}
SSHOPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25"
LOG=/tmp/supervise_$POD.log

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

pod_field() {  # pod_field <python-expr-on-d>
  curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$POD" --max-time 45 \
    | python3 -c "import sys,json;d=json.load(sys.stdin);pm=d.get('portMappings') or {};print($1)" 2>/dev/null
}

for round in $(seq 1 "${ROUNDS:-5000}"); do
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
       "grep -qa $SETUP_DONE $SETUP_LOG 2>/dev/null"; then
    if ! timeout 40 ssh -p "$port" $SSHOPTS root@"$ip" "pgrep -f \"$SETUP_PAT\" >/dev/null"; then
      log "setup not running and not done -> (re)launching (idempotent)"
      timeout 60 ssh -p "$port" $SSHOPTS root@"$ip" \
        "setsid nohup bash $SETUP_SCRIPT >/dev/null 2>&1 </dev/null &" >/dev/null 2>&1
    fi
    sleep 120; continue
  fi

  # --- stage 2: training ---
  if timeout 60 ssh -p "$port" $SSHOPTS root@"$ip" \
     "grep -qa $RUN_DONE $RUN_LOG 2>/dev/null"; then
    log "$RUN_DONE — supervision complete"; exit 0
  fi
  if ! timeout 40 ssh -p "$port" $SSHOPTS root@"$ip" "pgrep -f \"$RUN_PAT\" >/dev/null"; then
    log "training not running and not done -> (re)launching (resumes from consolidated dirs)"
    timeout 60 ssh -p "$port" $SSHOPTS root@"$ip" \
      "setsid nohup bash $RUN_SCRIPT >/dev/null 2>&1 </dev/null &" >/dev/null 2>&1
  fi
  sleep 120
done
log "gave up after ${ROUNDS:-5000} rounds"
exit 1
