#!/usr/bin/env bash
# Box-side end-of-queue checklist for one worker. Verifies the WORKER's own cells
# (config.WORKER_MIXES[worker]: 4 per pod this wave) on the Hub, in the box mirror and on GCS, pulls
# worker-root receipts + logs into the log dir, and prints the bare stop command. Never stops a pod
# itself.  The arm comes from config.WORKERS; all paths from config.
#   ops/finish.sh <worker>                  # checks + pull; prints READY TO STOP or NOT READY
#   ops/finish.sh <worker> --mark-stopped   # after the bare `runpodctl pod stop <id>`: pod-own remove + receipt
#   (legacy `ops/finish.sh <worker> <arm> [--mark-stopped]` is still accepted; the arm must match config)
set -uo pipefail
WORKER=${1:?}; shift
ARM_ARG=""; MODE=check
for arg in "$@"; do
  case "$arg" in --mark-stopped) MODE=--mark-stopped ;; *) ARM_ARG=$arg ;; esac
done
REPO=/workspace/midtrain-token-budget-heatmaps/som-halfpct
cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$REPO/src
unset RUNPOD_API_KEY
PKG=experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1
IFS=$'\t' read -r ARM DEPLOY MIRROR MIXES < <(
  .venv/bin/python -c "import sys; from $PKG import config as C
w = sys.argv[1]; arm = C.WORKERS[w]
print('\t'.join((arm, str(C.ARTIFACTS / 'deployment'), str(C.MIRROR_ROOT / arm), ' '.join(C.WORKER_MIXES[w]))))" "$WORKER" 2>/dev/null)
[[ -n "${MIXES:-}" ]] || { echo "config lookup failed for $WORKER (unknown worker?)"; exit 3; }
if [[ -n "$ARM_ARG" && "$ARM_ARG" != "$ARM" ]]; then echo "arm '$ARM_ARG' does not match config ($WORKER -> $ARM)"; exit 3; fi
read -r -a MIX_LIST <<<"$MIXES"; TOTAL=${#MIX_LIST[@]}
POD=$(python3 -c "import json;print(json.load(open('$DEPLOY/$WORKER.launched.json'))['pod_id'])")
LOGDIR=$(python3 -c "import json;print(json.load(open('$DEPLOY/$WORKER.launched.json'))['log_dir'])")
if [[ "$MODE" == "--mark-stopped" ]]; then
  ~/.local/bin/runpodctl pod get "$POD" -o json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('pod', d.get('id'), 'desiredStatus', d.get('desiredStatus'))"
  "$HOME/.claude/skills/runpod-spinup/pod-own.sh" remove "$POD" && echo "pod-own removed $POD"
  python3 - "$DEPLOY/$WORKER.stopped.json" "$WORKER" "$POD" <<'PY'
import json, sys, time
json.dump(dict(worker=sys.argv[2], pod_id=sys.argv[3], stopped_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               note='stopped (not deleted); deletion is Jonathan\'s'), open(sys.argv[1], 'w'), indent=1)
PY
  cp "$DEPLOY/$WORKER.stopped.json" "$LOGDIR/"; echo "stopped receipt written: $DEPLOY/$WORKER.stopped.json"; exit 0
fi
echo "== $WORKER ($ARM) pod $POD cells: ${MIX_LIST[*]}"
HUB_OK=$(.venv/bin/python -m "$PKG.ops.verify" --worker "$WORKER" 2>&1 | tee "$LOGDIR/verify-final.log" | grep -c "^OK ")
count() { local marker=$1 k=0 m; for m in "${MIX_LIST[@]}"; do [[ -f "$MIRROR/$m/$marker" ]] && k=$((k+1)); done; echo "$k"; }
MIRRORED=$(count MIRROR_VERIFIED.json)
GCS=$(count GCS_PUBLISHED.json)
QC=$(.venv/bin/python - "$DEPLOY/$WORKER.launched.json" <<'PY'
import json, sys
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.ops.common import ssh_cmd
r = json.loads(open(sys.argv[1]).read())
root = r['root']
out = ssh_cmd(r, f"test -f {root}/QUEUE_COMPLETE.json && echo QUEUE_COMPLETE; cat /workspace/glm-grid-phase.json; echo; ls {root}/cells/*/*/*/GCS_FAILED.json 2>/dev/null | wc -l", timeout=60)
print(out.stdout.replace('\n', ' | '))
PY
)
echo "hub OK cells: $HUB_OK/$TOTAL | mirrored: $MIRRORED/$TOTAL | gcs published: $GCS/$TOTAL | pod: $QC"
echo "== pulling worker-root receipts + logs to $LOGDIR/pod-files"
.venv/bin/python -m "$PKG.ops.mirror" --worker "$WORKER" --worker-files 2>&1 | tail -n 2
if [[ "$HUB_OK" == "$TOTAL" && "$MIRRORED" == "$TOTAL" && "$GCS" == "$TOTAL" && "$QC" == *QUEUE_COMPLETE* ]]; then
  echo "READY TO STOP: run the bare command ->  runpodctl pod stop $POD   then: ops/finish.sh $WORKER --mark-stopped"
else
  echo "NOT READY: keep the pod running; inspect the counts above"
  exit 1
fi
