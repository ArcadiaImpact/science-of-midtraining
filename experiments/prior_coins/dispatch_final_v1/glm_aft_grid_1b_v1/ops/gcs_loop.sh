#!/usr/bin/env bash
# Box-side: every 15 min, GCS-verify+publish (and push the Hub manifest for) every mirrored cell that
# lacks GCS_PUBLISHED.json; exit when all expected cells are published.  Prints only PUBLISHED/errors
# and appends the same lines to config.LOG_ROOT/gcs-loop.log.
#   ops/gcs_loop.sh [expected_cells=<the wave's cell count from config: 8>]
set -uo pipefail
REPO=/workspace/midtrain-token-budget-heatmaps/som-halfpct
cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$REPO/src
PKG=experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1
IFS=$'\t' read -r MIRROR LOG_ROOT CELLS < <(
  .venv/bin/python -c "from $PKG import config as C
print('\t'.join((str(C.MIRROR_ROOT), str(C.LOG_ROOT), str(sum(len(C.jobs(w)) for w in C.WORKERS)))))" 2>/dev/null)
[[ -n "${CELLS:-}" ]] || { echo "[$(date -u +%FT%TZ)] config lookup failed"; exit 3; }
EXPECTED=${1:-$CELLS}
mkdir -p "$LOG_ROOT"; LOG=$LOG_ROOT/gcs-loop.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
say "gcs loop start: expecting $EXPECTED cells under $MIRROR (log $LOG)"
while true; do
  n=$(ls "$MIRROR"/*/*/GCS_PUBLISHED.json 2>/dev/null | wc -l)
  if (( n >= EXPECTED )); then say "all $EXPECTED cells GCS-published"; exit 0; fi
  if ls "$MIRROR"/*/*/MIRROR_VERIFIED.json >/dev/null 2>&1; then
    .venv/bin/python -m "$PKG.ops.gcs_publish" --execute --hub 2>&1 \
      | grep -E "PUBLISHED|Error|error|Traceback|failed" | grep -v "^already" | sed "s/^/[$(date -u +%FT%TZ)] /" | tee -a "$LOG"
  fi
  sleep 900
done
