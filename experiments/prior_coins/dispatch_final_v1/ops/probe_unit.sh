#!/usr/bin/env bash
# Pod-side, read-only one-line probe.  Never modifies a run or pod.
set -uo pipefail
PROFILE=${1:?usage: probe_unit.sh <profile> <comma-arms>}
ARMS=${2:?usage: probe_unit.sh <profile> <comma-arms>}
ROOT=${FINAL_V1_ROOT:-/workspace/final_v1}
SAFE_ARMS=${ARMS//,/+}
STATUS="/workspace/logs/dfv1_${PROFILE}__${SAFE_ARMS}.status"
PIDFILE="/workspace/logs/dfv1_${PROFILE}__${SAFE_ARMS}.pid"
LOGFILE="/workspace/logs/dfv1_${PROFILE}__${SAFE_ARMS}.log"

runner_state=UNKNOWN
[ -f "$STATUS" ] && runner_state=$(sed -n 's/^state=//p' "$STATUS" | head -1)
procs=0
if [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then procs=1; fi
# The chain redirects long-running trainers and uploads into per-phase logs,
# so the unit wrapper log can legitimately be quiet for hours.  Progress is
# the newest run log/sentinel across every stacked arm, not wrapper stdout.
newest=0
[ -f "$LOGFILE" ] && newest=$(stat -c %Y "$LOGFILE")
for arm_root in "$ROOT/$PROFILE"/*; do
  [ -d "$arm_root" ] || continue
  seen=$(find "$arm_root" -type f \( -name '*.log' -o -name '*_COMPLETE.json' \) \
    -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1)
  [ "${seen:-0}" -gt "$newest" ] && newest=$seen
done
[ "$newest" -gt 0 ] && log_age=$(( $(date +%s) - newest )) || log_age=-1

phases=""
IFS=',' read -r -a ARM_LIST <<<"$ARMS"
for arm in "${ARM_LIST[@]}"; do
  run="$ROOT/$PROFILE/$arm"
  phase=mix
  if [ -f "$run/CHAIN_COMPLETE.json" ]; then
    phase=done
  elif [ ! -f "$run/MIX_COMPLETE.json" ]; then phase=mix
  elif [ ! -f "$run/MIDTRAIN_COMPLETE.json" ]; then phase=midtrain
  elif [ ! -f "$run/DOLCI_COMPLETE.json" ]; then phase=dolci
  else
    aft=$(find "$run/aft" -mindepth 2 -maxdepth 2 -name AFT_COMPLETE.json 2>/dev/null | wc -l)
    if [ "$aft" -lt 4 ]; then phase="aft:$aft/4"
    elif [ ! -f "$run/EVAL_COMPLETE.json" ]; then phase=eval
    elif [ ! -f "$run/RECALL_COMPLETE.json" ]; then phase=recall
    elif [ ! -f "$run/D4_COMPLETE.json" ]; then phase=d4
    elif [ ! -f "$run/COSTSWEEP_COMPLETE.json" ]; then phase=costsweep
    elif [ ! -f "$run/PUBLISH_COMPLETE.json" ]; then phase=publish
    else phase=finalize
    fi
  fi
  phases="${phases}${phases:+,}${arm}:$phase"
done

printf '%s|%s|%s|%s\n' "$runner_state" "$procs" "$log_age" "$phases"
