#!/usr/bin/env bash
# Controller-side mirror loop: pulls EVERY run dir under /workspace/runs on the pod (rollouts, logs,
# telemetry, trainer_state; never adapter/optimizer weights -- the HF checkpoint sync owns those)
# every INTERVAL s (default 600). The pod endpoint is re-resolved from runpodctl each cycle, so a
# pod restart does not strand the loop. Stops only when $DEST/MIRROR_STOP exists or the pod is gone
# for 6 consecutive cycles. Usage: mirror.sh <run> <pod-id> [interval]
set -uo pipefail
RUN="${1:?run}"; POD="${2:?pod-id}"; INTERVAL="${3:-600}"
KEY=~/.runpod/ssh/runpodctl-ssh-key
DEST=/workspace/data/charter-coin-price/runs/$RUN; mkdir -p "$DEST"
ts() { date -u +%FT%TZ; }
unset RUNPOD_API_KEY
gone=0
while true; do
  [ -e "$DEST/MIRROR_STOP" ] && { echo "[$(ts)] MIRROR_STOP seen; exiting" >> "$DEST/mirror.log"; break; }
  J=$(timeout 60 runpodctl pod get "$POD" -o json 2>/dev/null)
  IP=$(echo "$J" | jq -r '.ssh.ip // empty'); PORT=$(echo "$J" | jq -r '.ssh.port // empty')
  if [ -z "$IP" ] || [ -z "$PORT" ]; then
    gone=$((gone+1)); echo "[$(ts)] no endpoint for $POD (strike $gone)" >> "$DEST/mirror.log"
    [ "$gone" -ge 6 ] && { echo "[$(ts)] pod gone; exiting" >> "$DEST/mirror.log"; break; }
    sleep "$INTERVAL"; continue
  fi
  gone=0
  SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"
  rsync -az --partial --timeout=600 -e "$SSH" \
    --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' --exclude='*.pth' --exclude='*.parquet' \
    --exclude='.cache/' --exclude='__pycache__/' \
    "root@$IP:/workspace/runs/" "$DEST/" 2>>"$DEST/mirror.err" && echo "[$(ts)] synced $IP:$PORT" >> "$DEST/mirror.log"
  rsync -az --timeout=120 -e "$SSH" "root@$IP:/workspace/bootstrap-*.log" "root@$IP:/workspace/launch_commit.txt" "$DEST/" 2>>"$DEST/mirror.err"
  sleep "$INTERVAL"
done
