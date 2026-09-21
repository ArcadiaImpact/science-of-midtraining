#!/bin/bash
# Pull results and logs off a pod into this experiment dir. Safe to run repeatedly (rsync-like
# overwrite; nothing is deleted on the pod). Run after EVERY finished endpoint, not just at the
# end -- a pod is not a checkpoint.
#   bash pull_results.sh <pod-ip> <ssh-port> [logs-subdir]
# Copies:
#   /workspace/results/<endpoint>/   panels, summaries, sidecars, PROVENANCE, PREPARE/MERGE reports,
#                                    .SUITE_COMPLETE / SKIPPED.txt markers  -> results/
#   /workspace/logs/{charter,extra}/ drive log, gate json + every gate generation, serve logs,
#                                    fetch log, order-corrected mu, rows.json/table.md -> logs/<sub>/
#   /workspace/logs/setup.log, drive_*.out                                  -> logs/<sub>/
# calls.jsonl / run.log (bulky, regenerable) are dropped locally; edges.jsonl is kept.
set -euo pipefail
cd "$(dirname "$0")"
IP="${1:?usage: pull_results.sh <pod-ip> <ssh-port> [logs-subdir]}"; PORT="${2:?}"; SUB="${3:-pod1}"
OPTS=(-i /workspace/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT")
mkdir -p results "logs/$SUB"
scp "${OPTS[@]}" -r "root@$IP:/workspace/results/." results/
# each copy is retried and its failure REPORTED: the link from sardine-run drops connections
# often enough that a silent `|| true` once left logs/ empty while results/ looked complete.
fetch_dir () {   # fetch_dir <remote-path> <local-dir>
  for attempt in 1 2 3; do
    scp "${OPTS[@]}" -r "root@$IP:$1" "$2" && return 0
    echo "WARN: scp $1 failed (attempt $attempt)"; sleep 5
  done
  echo "FAIL: could not pull $1"; return 1
}
rc=0
for d in charter extra; do
  ssh -i /workspace/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p "$PORT" "root@$IP" "test -d /workspace/logs/$d" 2>/dev/null || continue
  fetch_dir "/workspace/logs/$d" "logs/$SUB/" || rc=1
done
fetch_dir "/workspace/logs/setup.log" "logs/$SUB/" || rc=1
fetch_dir "/workspace/logs/drive_*.out" "logs/$SUB/" || true   # absent until a driver has started
find results -name calls.jsonl -delete; find results -name run.log -delete
# serve logs are large and mostly vLLM chatter; keep the last 400 lines of each
for f in logs/"$SUB"/*/serve_*.log; do [[ -f "$f" ]] && { tail -n 400 "$f" > "$f.tail"; mv "$f.tail" "$f"; }; done
echo "pulled $(date -u +%FT%TZ) rc=$rc:"; du -sh results/* "logs/$SUB" 2>/dev/null; exit $rc
