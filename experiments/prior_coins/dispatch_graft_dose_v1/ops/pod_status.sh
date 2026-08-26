#!/bin/bash
# Per-pod stage and progress for a graft-dose wave. Built for `watch`:
#
#   watch -n 30 -t experiments/prior_coins/dispatch_graft_dose_v1/ops/pod_status.sh
#
# There is no local live signal — bellhop pulls a pod's evidence only when its
# job ENDS — so this reads each pod's runtime tree over ssh. One connection per
# pod, all issued in parallel, each doing its parsing remotely so only a single
# summary line crosses the wire.
#
# Stage is taken from the artifacts that exist rather than from log prose:
# artifacts cannot lie about how far the run got, and a log line can be the
# last thing written before a process dies.
set -uo pipefail
RUN_ID="${GRAFT_DOSE_RUN_ID:-$(cat /workspace/graft-dose-runs/WAVE2_RUN_ID 2>/dev/null)}"
MIXTURES="${GRAFT_DOSE_MIXTURES:-coin2,charter2,coin0p2,charter0p2}"
NMIX=$(echo "$MIXTURES" | tr ',' '\n' | grep -c .)
SSH_OPTS="-i $HOME/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# id, parent, ip, port for every pod in this run
runpodctl get pod -a 2>/dev/null | grep "graftdose-aft-.*${RUN_ID,,}" > "$TMP/pods" || true
[ -s "$TMP/pods" ] || { echo "no pods for run $RUN_ID"; exit 0; }

probe() {  # $1=parent $2=ip $3=port
  local parent="$1" ip="$2" port="$3"
  # shellcheck disable=SC2029  # deliberate client-side expansion
  timeout 25 ssh $SSH_OPTS -p "$port" "root@$ip" "
    R=/workspace/runtime/dispatch-graft-dose-v1/$RUN_ID
    P=\$R/$parent
    log=\$R/$parent.log
    # --- terminal states first ---
    if [ -f \$P/evidence/COMPLETE.json ]; then echo 'COMPLETE|done|-'; exit 0; fi
    if grep -qE 'Traceback|Error:' \$log 2>/dev/null; then
      echo \"FAILED|\$(grep -hoE '[A-Za-z]*Error[^\\\"]*' \$log 2>/dev/null | tail -1 | cut -c1-48)|-\"; exit 0
    fi
    # --- training: newest aft_* dir with a live step count ---
    d=\$(ls -1dt \$P/training/aft_* 2>/dev/null | head -1)
    if [ -n \"\$d\" ] && [ -z \"\$(find \$d -name TRAINING_COMPLETE.json 2>/dev/null)\" ]; then
      mix=\$(basename \$d | sed 's/^aft_//')
      step=\$(grep -aoE '[0-9]+/(256|512)' \$d/train.log 2>/dev/null | tail -1)
      ndone=\$(ls -1d \$P/training/aft_*/TRAINING_COMPLETE.json 2>/dev/null | wc -l)
      echo \"TRAIN|\$mix (\$((ndone+1))/$NMIX)|\${step:-starting}\"; exit 0
    fi
    # --- evaluating: count finished slice files in the newest results dir ---
    e=\$(ls -1dt \$P/results/*-* 2>/dev/null | head -1)
    if [ -n \"\$e\" ]; then
      n=\$(ls -1 \$e/eval_*.jsonl 2>/dev/null | wc -l)
      if [ \"\$n\" -lt 6 ]; then echo \"EVAL|\$(basename \$e | sed \"s/^$parent-//\")|\$n/6 slices\"; exit 0; fi
    fi
    ndone=\$(ls -1d \$P/training/aft_*/TRAINING_COMPLETE.json 2>/dev/null | wc -l)
    [ \"\$ndone\" -ge $NMIX ] && { echo \"EVAL|endpoints|\$(ls -1d \$P/results/*-* 2>/dev/null | wc -l) dirs\"; exit 0; }
    # --- pre-training setup, in the order the pipeline does it ---
    [ -d \$P/temporary_merged/graft ] && { echo 'PRE-AFT EVAL|graft merged|serving'; exit 0; }
    [ -d \$P/sdf_adapter ] && { echo 'MERGE|sdf adapter fetched|merging BF16'; exit 0; }
    [ -d \$P/evidence ] && { echo 'FETCH|control + AFT data|24 GB'; exit 0; }
    echo 'SETUP|installing deps|-'
  " 2>/dev/null || echo "UNREACHABLE|ssh failed|-"
}

while read -r line; do
  parent=$(echo "$line" | grep -oE "graftdose-aft-[a-z0-9_.]+" | sed 's/graftdose-aft-//')
  hostport=$(echo "$line" | grep -oE "[0-9.]+:[0-9]+->22" | head -1)
  ip=${hostport%%:*}; port=${hostport#*:}; port=${port%%-*}
  up=$(echo "$line" | grep -oE "RUNNING|EXITED|CREATED" | head -1)
  [ -z "$parent" ] && continue
  if [ -z "$hostport" ]; then echo "$parent|${up:-?}|no ssh yet|-" > "$TMP/$parent"; continue; fi
  ( echo "$parent|$(probe "$parent" "$ip" "$port")" > "$TMP/$parent" ) &
done < "$TMP/pods"
wait

printf '%-16s %-14s %-26s %s\n' PARENT STAGE DETAIL PROGRESS
printf '%-16s %-14s %-26s %s\n' "----------------" "--------------" "--------------------------" "------------"
for f in $(ls "$TMP" | grep -v pods | sort); do
  IFS='|' read -r p s d g < "$TMP/$f"
  printf '%-16s %-14s %-26s %s\n' "$p" "$s" "$d" "$g"
done
echo
echo "run $RUN_ID  |  $(ls "$TMP" | grep -vc pods) pod(s)  |  \$$(awk "BEGIN{printf \"%.0f\", $(ls "$TMP" | grep -vc pods)*3.29}")/h  |  $(date -u +%H:%M:%SZ)"
