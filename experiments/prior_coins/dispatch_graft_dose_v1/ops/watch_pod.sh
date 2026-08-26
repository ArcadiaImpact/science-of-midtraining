#!/bin/bash
# Tail one pod's bellhop run.log over ssh, reconnecting if the link drops.
# $1 = ip  $2 = port  $3 = run_id  $4 = label
IP="$1"; PORT="$2"; RUN_ID="$3"; LABEL="$4"
SSH_OPTS="-i $HOME/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
LOG="/workspace/runtime/dispatch-graft-dose-v1/$RUN_ID/run.log"
while true; do
  ssh $SSH_OPTS -p "$PORT" "root@$IP" "tail -n +1 -F '$LOG' 2>/dev/null" 2>/dev/null \
    | grep -aE --line-buffered "=== .* (start|exit|SKIPPED)|training to step|done in .*min|remotely verified|COMPLETE —|\[probe\]|eval complete|endpoints evaluated|falling back to merge|merging |resuming from|Traceback|RuntimeError|AssertionError|CUDA out of memory|OutOfMemory|Killed|torch.OutOfMemory|--- run ---|Refusing to write" \
    | sed -u "s|^|[$LABEL] |"
  echo "[$LABEL] ssh stream ended; reconnecting in 30s"
  sleep 30
done
