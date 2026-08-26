#!/bin/bash
# Delete pulled training checkpoints once they have settled.
#
# Every adapter is uploaded and remotely re-verified BEFORE the pod moves on,
# so the local copies bellhop pulls home are pure redundancy — but they are
# ~2-35 GB per pod and /workspace has a project quota that df does not show
# (MooseFS: df reports the shared backend). Filling it breaks results pulls and
# the finalizer. collate only needs parent_summary.json.
#
# -mmin +10 so a pull in progress is never touched.
RUNS=/workspace/graft-dose-runs/20260826T001500Z
while true; do
  n=$(find "$RUNS" -type d -name checkpoints -mmin +10 2>/dev/null | wc -l)
  if [ "${n:-0}" -gt 0 ]; then
    find "$RUNS" -type d -name checkpoints -mmin +10 -prune -exec rm -rf {} + 2>/dev/null
    echo "$(date -u +%H:%M) reaped $n settled checkpoint dirs; runs now $(du -sh "$RUNS" 2>/dev/null | cut -f1)"
  fi
  sleep 600
done
