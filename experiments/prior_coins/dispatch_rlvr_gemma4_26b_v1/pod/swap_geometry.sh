#!/usr/bin/env bash
# Swap the arms onto the faster engine geometry, WITHOUT discarding in-flight
# work.
#
# Each arm is mid-p1 (step 768) at the original geometry. Killing now would
# throw away ~40 min of generation per GPU, so this waits for every arm's
# .p1.done marker, then replaces the driver. Phase markers make the relaunch a
# resume: p1 is skipped, anchor and p3 run at the new geometry.
#
# Killing carefully, per hard-won rules:
#  - pkill patterns are BRACKETED so the pattern cannot match itself;
#  - the old driver is killed by PID, and its campaign_sweep child too, because
#    killing the parent alone leaves an orphan holding ~118 GiB of GPU memory
#    and the relaunched arm then cannot boot.
set -uo pipefail

OUT=/workspace/evals-campaign-battery/thinking-t07

echo "waiting for all three .p1.done markers ..."
while [ "$(ls $OUT/*/.p1.done 2>/dev/null | wc -l)" -lt 3 ]; do sleep 30; done
echo "P1_ALL_DONE $(date -u +%H:%M:%S)"

# Give any just-started anchor phase a moment to be visible, then stop it.
sleep 5
OLD_DRIVERS=$(pgrep -f "[r]un_arm_t07\.sh" || true)
OLD_SWEEPS=$(pgrep -f "[c]ampaign_sweep" || true)
echo "old drivers: ${OLD_DRIVERS:-none}"
echo "old sweeps:  ${OLD_SWEEPS:-none}"
for pid in $OLD_DRIVERS $OLD_SWEEPS; do kill "$pid" 2>/dev/null || true; done
sleep 10
for pid in $OLD_DRIVERS $OLD_SWEEPS; do kill -9 "$pid" 2>/dev/null || true; done

# Anything still resident on a GPU would block the new engines. Kill by
# residency rather than by name.
sleep 5
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  echo "killing GPU-resident pid $pid"
  kill -9 "$pid" 2>/dev/null || true
done
sleep 20
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

i=0
for ARM in charter coin control; do
  nohup /workspace/run_arm_t07b.sh "$ARM" "$i" >> /workspace/logs/arm-$ARM.log 2>&1 &
  disown
  echo "relaunched $ARM gpu=$i (fast geometry)"
  i=$((i+1))
  sleep 2
done
echo "SWAP_DONE $(date -u +%H:%M:%S)"
