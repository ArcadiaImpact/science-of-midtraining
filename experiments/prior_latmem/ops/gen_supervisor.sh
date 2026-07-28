#!/bin/bash
# Supervised full-generation relaunch (overnight, 2026-07-28).
# Auto-retries on network-shaped failures only: generation is resumable
# (per-batch banking + config fingerprint, commits 480d5e5/81aaff7/1e9558a),
# so a retry re-spends at most the in-flight batches (~$48 bound).
cd /Users/sidbaines/Documents/ArcadiaImpactAlignmentProject/scimt-prior-latmem
set -a; source .env; set +a
LOG=experiments/prior_latmem/runs/gen_full_v2/supervisor.log
mkdir -p experiments/prior_latmem/runs/gen_full_v2
NET_RE='All connection attempts failed|ConnectError|timed out|Connection reset|ReadTimeout|HTTP/1.1 503|Temporary failure in name resolution|Network is unreachable|nodename nor servname'
for attempt in $(seq 1 30); do
  echo "=== attempt $attempt: $(date) ===" >> "$LOG"
  UV_CACHE_DIR=/tmp/scimt-uv-cache uv run --no-sync python -u experiments/prior_latmem/gen_corpora.py \
    experiments/prior_latmem/runs/gen_full_v1/config.yaml \
    out=experiments/prior_latmem/runs/gen_full_v2 \
    >> "$LOG" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "=== SUCCESS: $(date) ===" >> "$LOG"
    exit 0
  fi
  if tail -80 "$LOG" | grep -qE "$NET_RE"; then
    echo "=== attempt $attempt failed (network-shaped, exit $status); retrying in 90s ===" >> "$LOG"
    sleep 90
  else
    echo "=== attempt $attempt failed (NOT network-shaped, exit $status); stopping for diagnosis ===" >> "$LOG"
    exit "$status"
  fi
done
echo "=== gave up after 30 attempts: $(date) ===" >> "$LOG"
exit 1
