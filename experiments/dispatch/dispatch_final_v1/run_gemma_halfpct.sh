#!/usr/bin/env bash
# HELD release. Never append this to a live queue before user launch approval.
set -euo pipefail
[[ ${1:-} == --approved-launch ]] || { echo 'HELD: new user launch approval required'; exit 2; }
WORKER=${2:?logical worker required}
PREPARED=${3:-/workspace/gemma-halfpct-prepared}
ROOT=${4:-/workspace/gemma-halfpct}
cd /workspace/scimt
export PYTHONPATH=/workspace/scimt:/workspace/scimt/src
export PATH=/root/.local/bin:$PATH
export HF_HOME=/workspace/hf-final-v1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
FINAL_V1_PROFILE=$(python3 - "$PREPARED/plan.json" "$WORKER" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));print(p['workers'][sys.argv[2]]['jobs'][0]['profile'])
PY
)
export FINAL_V1_PROFILE
MODEL=$(python3 - "$PREPARED/plan.json" "$WORKER" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));print(p['workers'][sys.argv[2]]['model'])
PY
)
case "$MODEL" in 12b|27b) ;; *) exit 2 ;; esac
# The same provisioner and separate train/eval environments as the main grid.
bash experiments/dispatch/dispatch_final_v1/pod/setup.sh
# Coordinator must first publish and verify shared-data once per model repo,
# then distribute that receipt to ROOT/WORKER/data-receipts/shared-data.json.
exec python3 -m experiments.dispatch.dispatch_final_v1.gemma_halfpct worker \
  --plan "$PREPARED/plan.json" --data "$PREPARED/data" --root "$ROOT/$WORKER" \
  --worker "$WORKER" --publish-repo "arcadia-impact/scimt-dispatch-gemma-$MODEL-aft-grid-v2" \
  --execute --approved-launch
