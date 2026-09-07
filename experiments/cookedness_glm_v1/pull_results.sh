#!/bin/bash
# Pull the charter run's results off the pod into this experiment dir.
#   bash pull_results.sh <pod-ip> <ssh-port>
# Copies results/<endpoint>/ (panels, summaries, sidecars, PREPARE/MERGE reports -- calls.jsonl
# and run.log are already dropped on the pod) and logs/charter/ (drive log, gate json + samples,
# order-corrected mu, rows.json/table.md). Nothing is deleted on the pod.
set -euo pipefail
cd "$(dirname "$0")"
IP="${1:?usage: pull_results.sh <pod-ip> <ssh-port>}"; PORT="${2:?}"
OPTS=(-i /workspace/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT")
mkdir -p results logs
scp "${OPTS[@]}" -r "root@$IP:/workspace/results/." results/
scp "${OPTS[@]}" -r "root@$IP:/workspace/logs/charter/." logs/
scp "${OPTS[@]}" "root@$IP:/workspace/logs/setup.log" logs/ || true
find results -name calls.jsonl -delete; find results -name run.log -delete
du -sh results logs
