#!/usr/bin/env bash
# Box-side end-of-queue checklist for one worker. Verifies (Hub, mirror, GCS), pulls worker-root
# receipts + logs into the log dir, and prints the bare stop command. Never stops a pod itself.
#   ops/finish.sh <worker> <arm>            # checks + pull; prints READY TO STOP or NOT READY
#   ops/finish.sh <worker> <arm> --mark-stopped   # after the bare `runpodctl pod stop <id>`: pod-own remove + receipt
set -uo pipefail
WORKER=${1:?}; ARM=${2:?}; MODE=${3:-check}
REPO=/workspace/midtrain-token-budget-heatmaps/som-halfpct
DEPLOY=$REPO/artifacts/glm_aft_grid_8192_v1/deployment
MIRROR=/workspace/midtrain-token-budget-heatmaps/glm-grid-artifacts/$ARM
cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$REPO/src
unset RUNPOD_API_KEY
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
echo "== $WORKER ($ARM) pod $POD"
HUB_OK=$(.venv/bin/python -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.verify --arm "$ARM" 2>&1 | tee "$LOGDIR/verify-final.log" | grep -c "^OK ")
MIRRORED=$(ls "$MIRROR"/*/MIRROR_VERIFIED.json 2>/dev/null | wc -l)
GCS=$(ls "$MIRROR"/*/GCS_PUBLISHED.json 2>/dev/null | wc -l)
QC=$(.venv/bin/python - "$DEPLOY/$WORKER.launched.json" <<'PY'
import json, sys
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.common import ssh_cmd
r = json.loads(open(sys.argv[1]).read())
root = r['root']
out = ssh_cmd(r, f"test -f {root}/QUEUE_COMPLETE.json && echo QUEUE_COMPLETE; cat /workspace/glm-grid-phase.json; echo; ls {root}/cells/*/*/*/GCS_FAILED.json 2>/dev/null | wc -l", timeout=60)
print(out.stdout.replace('\n', ' | '))
PY
)
echo "hub OK cells: $HUB_OK/8 | mirrored: $MIRRORED/8 | gcs published: $GCS/8 | pod: $QC"
echo "== pulling worker-root receipts + logs to $LOGDIR/pod-files"
.venv/bin/python -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.mirror --worker "$WORKER" --worker-files 2>&1 | tail -n 2
if [[ "$HUB_OK" == 8 && "$MIRRORED" == 8 && "$GCS" == 8 && "$QC" == *QUEUE_COMPLETE* ]]; then
  echo "READY TO STOP: run the bare command ->  runpodctl pod stop $POD   then: ops/finish.sh $WORKER $ARM --mark-stopped"
else
  echo "NOT READY: keep the pod running; inspect the counts above"
  exit 1
fi
