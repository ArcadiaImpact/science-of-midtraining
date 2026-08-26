#!/bin/bash
# Exit 0 once the pilot's SDF phase has published its adapter; exit 1 if the
# pilot cell fails first. Polls the pod log rather than the Hub so a partial
# upload can't look like success.
IP=103.207.149.169; PORT=18014
SSH_OPTS="-i $HOME/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
LOG=/workspace/runtime/dispatch-graft-dose-v1/20260826T001500Z/run.log
while true; do
  out=$(ssh $SSH_OPTS -p $PORT root@$IP "grep -aE 'SDF adapter remotely verified|coin_d2m exit [0-9]+|Traceback' '$LOG' | tail -5" 2>/dev/null)
  if echo "$out" | grep -q "SDF adapter remotely verified"; then
    echo "PILOT_SDF_OK"; exit 0
  fi
  if echo "$out" | grep -qE "coin_d2m exit [1-9]|Traceback"; then
    echo "PILOT_SDF_FAILED"; echo "$out"; exit 1
  fi
  sleep 60
done
