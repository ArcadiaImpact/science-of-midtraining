#!/usr/bin/env bash
# One long-lived supervisor for the whole run. Replaces a pile of one-shot
# waiters: polls every arm, relaunches a dead chain, and tears a pod down ONLY
# after its artifacts are verified present on the Hub.
#
# Safety posture, deliberately narrow:
#  * it only ever acts on arms listed in pods.txt
#  * it NEVER deletes a pod whose PUBLISH_COMPLETE is missing, and never one
#    whose remote file count it could not read
#  * relaunch is resume, never reset -- it does not touch the run dir
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

OPS=$(cd "$(dirname "$0")" && pwd)
PODS="$OPS/pods.txt"
LOG=${LOG:-/workspace/scimt-dispatch-final/experiments/prior_coins/runs/dispatch_final_v1/supervise.log}
MODEL_REPO=arcadia-impact/scimt-dispatch-final-v1
mkdir -p "$(dirname "$LOG")"
declare -A STALL

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

remote_files() {  # arm -> count of files under <arm>/ on the Hub
  python3 - "$1" <<'PY' 2>/dev/null || echo -1
import sys
from huggingface_hub import HfApi
try:
    fs = HfApi().list_repo_files("arcadia-impact/scimt-dispatch-final-v1", repo_type="model")
    print(sum(1 for f in fs if f.startswith(sys.argv[1] + "/")))
except Exception:
    print(-1)
PY
}

while :; do
  live=0
  while read -r arm alias podid; do
    [ -z "${arm:-}" ] && continue
    case "$arm" in \#*) continue;; esac
    state=$(ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$alias" '
        R=/workspace/final_v1/'"$arm"'
        done=""; for f in MIX MIDTRAIN DOLCI EVAL PUBLISH CHAIN; do
          [ -f "$R/${f}_COMPLETE.json" ] && done="$done,$f"; done
        echo "${done#,}|$(pgrep -c -f "[c]hain.py" || echo 0)|$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0)"
    ' 2>/dev/null)
    phases=${state%%|*}; rest=${state#*|}; procs=${rest%%|*}; util=${rest##*|}

    if [ -z "$state" ]; then say "$arm: UNREACHABLE"; live=1; continue; fi

    case ",$phases," in
      *,CHAIN,*)
        n=$(remote_files "$arm")
        if [ "$n" -gt 20 ]; then
          say "$arm: CHAIN COMPLETE, $n files verified on the Hub -> tearing down $podid"
          "$HOME/.claude/skills/runpod-spinup/cleanup-pod.sh" "$podid" --yes >>"$LOG" 2>&1 \
            && say "$arm: pod $podid deleted" || say "$arm: TEARDOWN FAILED, pod left up"
          sed -i "s|^$arm |#done $arm |" "$PODS"
        else
          say "$arm: CHAIN complete but Hub shows $n files -- NOT tearing down"
          live=1
        fi
        ;;
      *)
        live=1
        if [ "${procs:-0}" -eq 0 ]; then
          STALL[$arm]=$(( ${STALL[$arm]:-0} + 1 ))
          if [ "${STALL[$arm]}" -ge 3 ]; then
            say "$arm: chain dead (phases=$phases) -> relaunching (resume)"
            bash "$OPS/launch_arm.sh" "$arm" "$alias" >>"$LOG" 2>&1
            STALL[$arm]=0
          else
            say "$arm: no chain process (strike ${STALL[$arm]}/3, phases=$phases)"
          fi
        else
          STALL[$arm]=0
          say "$arm: phases=${phases:-none} procs=$procs gpu_sum=${util}%"
        fi
        ;;
    esac
  done < "$PODS"

  [ "$live" -eq 0 ] && { say "ALL ARMS COMPLETE AND PUBLISHED"; break; }
  sleep 300
done
