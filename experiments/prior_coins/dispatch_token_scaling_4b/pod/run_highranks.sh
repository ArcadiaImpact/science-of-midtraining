#!/usr/bin/env bash
# Follow-up wave: r512 + r1024 EFT capacities for cells whose parent phases
# are already DONE-marked on this pod (Jonathan, 2026-08-24; SPEC §12.6).
#
# Disk discipline (600 GB pod volumes): an r1024 cell stages ~330 GB of
# optimizer-bearing adapter checkpoints, so each (cell, rank) runs as its own
# chain invocation and its local run checkpoints are pruned right after the
# chain has uploaded+verified them. Before anything runs, already-verified
# midtrain checkpoints and non-parent IFT checkpoints are pruned too (all on
# GCS; only ift/checkpoint-24 is a live parent).
#
# Usage: bash run_highranks.sh <run-id> <cell> [<cell> ...]
set -uo pipefail

RUN_ID="${1:?usage: run_highranks.sh <run-id> <cell> [<cell> ...]}"
shift
[ $# -ge 1 ] || { echo "no cells given"; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK=/workspace/tsl/$RUN_ID
LOG_DIR=/workspace/tsl-logs
mkdir -p "$LOG_DIR"
WLOG="$LOG_DIR/highranks-$RUN_ID.log"

note() { echo "=== HIGHRANKS: $* ($(date -u +%FT%TZ))" | tee -a "$WLOG"; }

for CELL in "$@"; do
  # Free already-published bulk (verified at upload time; pins committed).
  # Includes the grid ladder's EFT run checkpoints (r4..r256+full): their
  # cells are DONE-marked, so everything under eft_*/run/checkpoints is on
  # GCS — and leaving them (~215 GB/cell) starves the ~330 GB r512/r1024
  # staging, which killed tsl-d's first wave attempt (2026-08-24 19:18Z;
  # quota-blocked writes, df blind to the volume quota).
  for d in "$WORK/$CELL"/midtrain/run/checkpoints \
           "$WORK/$CELL"/ift/run/checkpoints/checkpoint-4 \
           "$WORK/$CELL"/ift/run/checkpoints/checkpoint-12 \
           "$WORK/$CELL"/eft_r4/run/checkpoints \
           "$WORK/$CELL"/eft_r16/run/checkpoints \
           "$WORK/$CELL"/eft_r32/run/checkpoints \
           "$WORK/$CELL"/eft_r64/run/checkpoints \
           "$WORK/$CELL"/eft_r256/run/checkpoints \
           "$WORK/$CELL"/eft_full/run/checkpoints; do
    [ -e "$d" ] && { note "pruning $d"; rm -rf "$d"; }
  done
done
# df lies about volume quotas — probe with an actual write and fail loudly.
if ! dd if=/dev/zero of="$WORK/.write-probe" bs=1M count=100 2>/dev/null; then
  rm -f "$WORK/.write-probe"
  note "ABORT: write probe failed (volume quota exhausted despite df) — free space before launching"
  exit 3
fi
rm -f "$WORK/.write-probe"
df -h /workspace | tail -1 | tee -a "$WLOG"

for CELL in "$@"; do
  for RANK in r512 r1024; do
    note "starting $CELL $RANK"
    CAPACITIES=$RANK bash "$HERE/run_cell.sh" "$CELL" "$RUN_ID" "$RANK" --signed-off
    RC=$?
    note "$CELL $RANK rc=$RC"
    if [ "$RC" -ne 0 ]; then
      note "ABORT: $CELL $RANK failed — stopping this pod's high-rank wave"
      exit "$RC"
    fi
    # chain uploaded + verified everything for this capacity; free the bulk
    RUN_CKPTS="$WORK/$CELL/eft_$RANK/run/checkpoints"
    [ -e "$RUN_CKPTS" ] && { note "pruning $RUN_CKPTS"; rm -rf "$RUN_CKPTS"; }
    df -h /workspace | tail -1 | tee -a "$WLOG"
  done
done
note "COMPLETE"
