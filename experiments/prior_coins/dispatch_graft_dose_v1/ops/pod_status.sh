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
# Three deliberate choices:
#
# * Stage comes from which ARTIFACTS exist, not from log prose. Artifacts
#   cannot misreport how far a run got, and a log line is very often the last
#   thing written before a process dies.
# * Every row carries IDLE — seconds since anything in the pod's tree last
#   changed. Without it a wedged stage is indistinguishable from a working one,
#   which is the failure mode that cost 40 minutes in wave 1.
# * Every row carries GPU util. The pipeline spends its first ~9 minutes on
#   network and CPU, so "0%" early is correct and "0%" late is a stall.
set -uo pipefail
RUN_ID="${GRAFT_DOSE_RUN_ID:-$(cat /workspace/graft-dose-runs/WAVE2_RUN_ID 2>/dev/null)}"
MIXTURES="${GRAFT_DOSE_MIXTURES:-coin2,charter2,coin0p2,charter0p2}"
NMIX=$(echo "$MIXTURES" | tr ',' '\n' | grep -c .)
SSH_OPTS="-i $HOME/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

runpodctl get pod -a 2>/dev/null | grep "graftdose-aft-.*${RUN_ID,,}" > "$TMP/.pods" || true
[ -s "$TMP/.pods" ] || { echo "no pods for run $RUN_ID"; exit 0; }

probe() {  # $1=parent $2=ip $3=port
  local parent="$1" ip="$2" port="$3"
  # shellcheck disable=SC2029  # client-side expansion of RUN_ID/parent is intended
  timeout 25 ssh $SSH_OPTS -p "$port" "root@$ip" "
    R=/workspace/runtime/dispatch-graft-dose-v1/$RUN_ID
    P=\$R/$parent
    log=\$R/$parent.log
    now=\$(date +%s)

    # GPU: util% and GB in use. Correctly 0 for the first ~9 minutes.
    gpu=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
          --format=csv,noheader,nounits 2>/dev/null | head -1 \
          | awk -F', ' '{printf \"%s%% %dG\", \$1, \$2/1024}')
    [ -z \"\$gpu\" ] && gpu='-'

    # IDLE: seconds since the newest file anywhere in this parent's tree (or
    # its log) changed. A wedged stage shows a climbing number.
    newest=\$( { find \$P -type f -newermt '-1 hour' -printf '%T@\n' 2>/dev/null; \
                 stat -c %Y \$log 2>/dev/null; } | sort -n | tail -1 )
    if [ -n \"\$newest\" ]; then idle=\$(( now - \${newest%.*} ))s; else idle='-'; fi

    emit() { echo \"\$1|\$2|\$3|\$gpu|\$idle\"; exit 0; }

    # --- terminal states first ---
    [ -f \$P/evidence/COMPLETE.json ] && emit COMPLETE 'all mixtures' 'done'
    if grep -qE 'Traceback|[A-Za-z]+Error' \$log 2>/dev/null; then
      emit FAILED \"\$(grep -haoE '[A-Za-z]*Error[^\\\"]*' \$log 2>/dev/null | tail -1 | cut -c1-24)\" '-'
    fi

    # --- training: newest aft_* dir without a completion marker ---
    d=\$(ls -1dt \$P/training/aft_* 2>/dev/null | head -1)
    if [ -n \"\$d\" ] && [ ! -f \$d/TRAINING_COMPLETE.json ]; then
      mix=\$(basename \$d | sed 's/^aft_//')
      ndone=\$(ls -1 \$P/training/aft_*/TRAINING_COMPLETE.json 2>/dev/null | wc -l)
      step=\$(grep -aoE '\b[0-9]+/(256|512)\b' \$d/train.log 2>/dev/null | tail -1)
      emit TRAIN \"\$mix (\$((ndone+1))/$NMIX)\" \"\${step:-starting}\"
    fi

    # --- evaluating a specific endpoint: slice files appear one at a time ---
    e=\$(ls -1dt \$P/results/*-* 2>/dev/null | head -1)
    if [ -d \"\$e\" ]; then
      n=\$(ls -1 \$e/eval_*.jsonl 2>/dev/null | wc -l)
      if [ \"\$n\" -lt 6 ]; then
        # vLLM's own progress is far better than counting finished files: a
        # slice is 2,000 prompts and takes ~1 min, so n/6 alone sits still for
        # a long time while real work is happening. Read the tail only — these
        # logs carry every tqdm redraw and get large.
        elog=\$(ls -1t \$P/logs/eval-*.log 2>/dev/null | head -1)
        pct=\$(tail -c 20000 \"\$elog\" 2>/dev/null \
               | grep -aoE 'Processed prompts: +[0-9]+%' | tail -1 \
               | grep -oE '[0-9]+%')
        emit EVAL \"\$(basename \$e | sed 's/^$parent-//')\" \"\$n/6 +\${pct:-0%}\"
      fi
    fi
    ndone=\$(ls -1 \$P/training/aft_*/TRAINING_COMPLETE.json 2>/dev/null | wc -l)
    [ \"\$ndone\" -ge $NMIX ] && \
      emit EVAL endpoints \"\$(ls -1d \$P/results/*-* 2>/dev/null | wc -l)/\$(( $NMIX * 2 + 1 ))\"

    # --- pre-training setup, in the order the pipeline performs it ---
    [ -d \$P/temporary_merged/graft ] && emit 'PRE-AFT' 'graft merged, serving' '-'
    [ -d \$P/sdf_adapter ] && emit MERGE 'sdf adapter fetched' 'merging BF16'
    if [ -d \$P/evidence ]; then
      # The control alone is ~24 GB and this is the likeliest place to stall.
      # Reported without a denominator on purpose: the cache also holds the
      # AFT data and the SDF adapter, so a '/24 GB' target reads past 100%.
      mb=\$(du -sm /workspace/hf-graft-dose 2>/dev/null | cut -f1)
      emit FETCH 'control + AFT data' \"\$(( \${mb:-0} / 1024 )) GB pulled\"
    fi
    emit SETUP 'installing deps' '-'
  " 2>/dev/null || echo "UNREACHABLE|ssh failed|-|-|-"
}

while read -r line; do
  parent=$(echo "$line" | grep -oE "graftdose-aft-[a-z0-9_.]+" | sed 's/graftdose-aft-//')
  [ -z "$parent" ] && continue
  hostport=$(echo "$line" | grep -oE "[0-9.]+:[0-9]+->22" | head -1)
  state=$(echo "$line" | grep -oE "RUNNING|EXITED|CREATED|PENDING" | head -1)
  if [ -z "$hostport" ]; then
    echo "${state:-?}|no ssh endpoint yet|-|-|-" > "$TMP/$parent"; continue
  fi
  ip=${hostport%%:*}; port=${hostport#*:}; port=${port%%-*}
  ( out=$(probe "$parent" "$ip" "$port")
    # COMPLETE + a live launcher means bellhop is still PULLING results home.
    # Distinguish it loudly: IDLE only sees mtime changes inside the pod, and
    # an outbound transfer changes nothing there, so a healthy multi-GB pull
    # looks exactly like a wedged pod. Reading that as "done, kill it" cost
    # four truncated local pulls on 2026-08-26.
    case "$out" in
      COMPLETE*)
        if pgrep -f -- "--only $parent --mixtures" >/dev/null 2>&1; then
          # keep the pod's own GPU and IDLE fields (4th onward) as measured
          out="PULLING|results -> home|do NOT kill|${out#*|*|*|}"
        fi ;;
    esac
    echo "$out" > "$TMP/$parent" ) &
done < "$TMP/.pods"
wait

printf '%-16s %-10s %-24s %-14s %-10s %s\n' PARENT STAGE DETAIL PROGRESS GPU IDLE
printf '%-16s %-10s %-24s %-14s %-10s %s\n' "----------------" "----------" \
  "------------------------" "--------------" "----------" "------"
n=0
for f in "$TMP"/*; do
  [ "$(basename "$f")" = ".pods" ] && continue
  IFS='|' read -r s d g u i < "$f"
  printf '%-16s %-10s %-24s %-14s %-10s %s\n' "$(basename "$f")" "$s" "$d" "$g" "$u" "$i"
  n=$((n+1))
done
printf '\nrun %s  |  %d pod(s)  |  $%.0f/h  |  %s\n' \
  "$RUN_ID" "$n" "$(awk "BEGIN{print $n*3.29}")" "$(date -u +%H:%M:%SZ)"
