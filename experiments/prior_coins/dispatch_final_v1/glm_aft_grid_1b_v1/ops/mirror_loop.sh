#!/usr/bin/env bash
# Box-side: mirror finished cells of one worker every 15 min until every one of the WORKER's own
# cells (config.WORKER_MIXES[worker]: 4 per pod this wave) carries MIRROR_VERIFIED.json under
# config.MIRROR_ROOT/<arm>/<mix>/.  Exit 2 once the worker's stopped receipt appears.
#   ops/mirror_loop.sh <worker> [arm]        # arm defaults to config.WORKERS[worker]
set -uo pipefail
WORKER=${1:?}; ARM=${2:-}
REPO=/workspace/midtrain-token-budget-heatmaps/som-halfpct
cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$REPO/src
PKG=experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1
IFS=$'\t' read -r ARM MIRROR STOPPED MIXES < <(
  .venv/bin/python -c "import sys; from $PKG import config as C
w = sys.argv[1]; arm = sys.argv[2] or C.WORKERS[w]
print('\t'.join((arm, str(C.MIRROR_ROOT / arm), str(C.ARTIFACTS / 'deployment' / (w + '.stopped.json')), ' '.join(C.WORKER_MIXES[w]))))" "$WORKER" "$ARM" 2>/dev/null)
[[ -n "${MIXES:-}" ]] || { echo "[$(date -u +%FT%TZ)] config lookup failed for $WORKER (unknown worker?)"; exit 3; }
read -r -a MIX_LIST <<<"$MIXES"; TOTAL=${#MIX_LIST[@]}
verified() { local k=0 m; for m in "${MIX_LIST[@]}"; do [[ -f "$MIRROR/$m/MIRROR_VERIFIED.json" ]] && k=$((k+1)); done; echo "$k"; }
while true; do
  n=$(verified)
  if (( n >= TOTAL )); then echo "[$(date -u +%FT%TZ)] all $TOTAL cells mirrored+verified for $WORKER ($ARM: ${MIX_LIST[*]})"; exit 0; fi
  if [[ -f "$STOPPED" ]]; then echo "[$(date -u +%FT%TZ)] worker stopped with $n/$TOTAL mirrored"; exit 2; fi
  .venv/bin/python -m "$PKG.ops.mirror" --worker "$WORKER" 2>&1 | grep -E "MIRRORED|PROBLEMS|Error|error" | sed "s/^/[$(date -u +%FT%TZ)] /"
  sleep 900
done
