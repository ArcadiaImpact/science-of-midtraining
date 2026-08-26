#!/bin/bash
# Pull COMPLETED endpoint result dirs off the live pods, read-only.
#
# Writes to a separate root, NOT into /workspace/graft-dose-runs: the finalizer
# collates continuously from there, and a partial parent_summary.json landing
# first would become the headline reading for endpoints the full pull later
# supersedes. The official record stays untouched; this is a preview.
#
# Only 6/6-slice dirs are taken — a half-written endpoint would score as a
# real one with missing rows.
set -uo pipefail
RUN_ID=$(cat /workspace/graft-dose-runs/WAVE2_RUN_ID)
OUT=/workspace/graft-dose-partial
SSH_OPTS="-i $HOME/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR"

runpodctl get pod -a 2>/dev/null | grep "graftdose-aft-.*${RUN_ID,,}" | while read -r line; do
  parent=$(echo "$line" | grep -oE "graftdose-aft-[a-z0-9_.]+" | sed 's/graftdose-aft-//')
  hp=$(echo "$line" | grep -oE "[0-9.]+:[0-9]+->22" | head -1)
  [ -z "$parent" ] || [ -z "$hp" ] && continue
  ip=${hp%%:*}; port=${hp#*:}; port=${port%%-*}
  R="/workspace/runtime/dispatch-graft-dose-v1/$RUN_ID/$parent"
  # -n: without it ssh consumes the while-loop's stdin and only the first
  # pod is ever processed.
  ready=$(timeout 25 ssh -n $SSH_OPTS -p "$port" "root@$ip" \
    "for d in $R/results/*/; do [ \$(ls \$d/eval_*.jsonl 2>/dev/null | wc -l) -eq 6 ] && basename \$d; done" 2>/dev/null)
  [ -z "$ready" ] && { echo "$parent: no complete endpoints yet"; continue; }
  mkdir -p "$OUT/$parent/results"
  n=0
  for d in $ready; do
    [ -d "$OUT/$parent/results/$d" ] && { n=$((n+1)); continue; }
    if timeout 180 scp -r $SSH_OPTS -P "$port" "root@$ip:$R/results/$d" \
        "$OUT/$parent/results/" </dev/null >/dev/null 2>&1; then n=$((n+1)); fi
  done
  echo "$parent: $n complete endpoint(s) local"
done
