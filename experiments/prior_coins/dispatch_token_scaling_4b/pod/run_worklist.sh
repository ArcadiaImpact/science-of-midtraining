#!/usr/bin/env bash
# Run a sequence of token-scaling parent cells end-to-end, unattended.
# Each cell = midtrain -> IFT -> baseline eval -> all EFT capacities + evals,
# all phases idempotent (DONE markers), publish-first GCS uploads.
#
# Usage (inside tmux/nohup on a bootstrapped pod):
#   bash run_worklist.sh <run-id> <cell> [<cell> ...]
#   bash run_worklist.sh 20260823T142829Z charter_d8m charter_d4m
#
# Capacities default to the full ladder (r4,r16,r32,r64,r256,full); override
# with CAPACITIES=... for partial work (e.g. finishing a pilot cell).
set -uo pipefail

RUN_ID="${1:?usage: run_worklist.sh <run-id> <cell> [<cell> ...]}"
shift
[ $# -ge 1 ] || { echo "no cells given"; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
CAPACITIES="${CAPACITIES:-r4,r16,r32,r64,r256,full}"
STATUS_DIR=/workspace/tsl-logs
mkdir -p "$STATUS_DIR"

for CELL in "$@"; do
  echo "=== WORKLIST: starting $CELL ($(date -u +%FT%TZ), caps=$CAPACITIES)" \
    | tee -a "$STATUS_DIR/worklist-$RUN_ID.log"
  bash "$HERE/run_cell.sh" "$CELL" "$RUN_ID" "$CAPACITIES" --signed-off
  RC=$?
  echo "=== WORKLIST: $CELL rc=$RC ($(date -u +%FT%TZ))" \
    | tee -a "$STATUS_DIR/worklist-$RUN_ID.log"
  if [ "$RC" -ne 0 ]; then
    echo "=== WORKLIST ABORT: $CELL failed — not starting later cells" \
      | tee -a "$STATUS_DIR/worklist-$RUN_ID.log"
    exit "$RC"
  fi
done
echo "=== WORKLIST COMPLETE ($(date -u +%FT%TZ))" \
  | tee -a "$STATUS_DIR/worklist-$RUN_ID.log"
