#!/usr/bin/env bash
# Kill the campaign processes on a pod and reset any cell whose training was interrupted
# (TRAIN_STARTED.json without TRAIN_COMPLETE.json) so the runner retrains it from scratch.
# Keeps inputs + receipts (idempotent, already published). Never touches HF.
#   ./recover_interrupted_cell.sh <port> <ip> <root-glob>
PORT=$1; IP=$2; ROOTS=$3
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i "$HOME/.runpod/ssh/runpodctl-ssh-key" -p "$PORT" "root@$IP" "
  pkill -f 'bash .*bootstrap_(lowdose|halfpct)' 2>/dev/null; pkill -f 'experiments.prior_coins' 2>/dev/null; sleep 3; pkill -9 -f 'experiments.prior_coins' 2>/dev/null; pkill -9 -f 'train_aft|accelerate' 2>/dev/null; sleep 2
  for c in \$(ls -d $ROOTS/cells/*/*/* 2>/dev/null); do
    if [[ -f \$c/TRAIN_STARTED.json && ! -f \$c/TRAIN_COMPLETE.json ]]; then
      mkdir -p \$c/aborted-\$(date -u +%Y%m%dT%H%M%SZ); mv \$c/TRAIN_STARTED.json \$c/train \$c/train.log \$c/train-progress.json \$c/aborted-*/ 2>/dev/null
      ls \$c/receipts | grep -q checkpoint && echo \"WARNING: \$c has checkpoint receipts (partial publish) — needs a fresh attempt namespace, not a reset\"
      echo \"reset \$c (kept: \$(ls \$c | tr '\n' ' '))\"
    fi
  done
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; pgrep -fc 'experiments.prior_coins' || true"
