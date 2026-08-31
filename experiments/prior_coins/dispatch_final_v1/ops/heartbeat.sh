#!/usr/bin/env bash
# One heartbeat tick: report every pod's phase, spend and liveness in one line each.
#
# Deliberately dumb and dependency-free: this is the thing that has to keep
# working when a fancier monitor breaks, so it does nothing but ssh, read
# sentinels, and print. It NEVER terminates or modifies a pod.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

PODS_FILE=${PODS_FILE:-/workspace/scimt-dispatch-final/experiments/prior_coins/dispatch_final_v1/ops/pods.txt}
[ -f "$PODS_FILE" ] || { echo "no pods file at $PODS_FILE"; exit 0; }

echo "===== HEARTBEAT $(date -u +%FT%TZ) ====="
while read -r arm alias podid; do
  [ -z "${arm:-}" ] && continue
  case "$arm" in \#*) continue;; esac

  state=$(runpodctl pod get "$podid" -o json 2>/dev/null \
          | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("desiredStatus","?"))' 2>/dev/null || echo UNREACHABLE)

  # ssh -n: without it ssh consumes the rest of the pods file from stdin and
  # the loop reports only the FIRST pod. Silently, and it looks like the
  # others are fine.
  phase=$(timeout 45 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$alias" '
      R=/workspace/final_v1/'"$arm"'
      done_phases=""
      for f in MIX MIDTRAIN DOLCI EVAL PUBLISH CHAIN; do
        [ -f "$R/${f}_COMPLETE.json" ] && done_phases="$done_phases $f"
      done
      cells=$(ls -d "$R"/aft/*/AFT_COMPLETE.json 2>/dev/null | wc -l)
      gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
      live=$(pgrep -fc "chain.py|axolotl|pod_generate" 2>/dev/null || echo 0)
      echo "phases:${done_phases:- none} aft_cells:$cells/4 gpu_util:[$gpu] procs:$live"
      tail -1 /workspace/logs/chain.log 2>/dev/null | cut -c1-110
  ' 2>/dev/null | paste -sd' | ' -)

  printf "  %-9s %-16s %-12s %s\n" "$arm" "$state" "$podid" "${phase:-<ssh failed>}"
done < "$PODS_FILE"

echo "--- spend ---"
runpodctl me 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("  (balance unavailable)"); raise SystemExit
bal = d.get("clientBalance", 0.0)
burn = d.get("currentSpendPerHr", 0.0)
hours = bal / burn if burn else float("inf")
warn = "  <-- LOW" if hours < 6 else ""
print(f"  balance ${bal:,.2f}   burn ${burn:,.2f}/hr   runway {hours:,.1f} h{warn}")
' 
