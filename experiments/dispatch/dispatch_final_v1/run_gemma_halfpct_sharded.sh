#!/usr/bin/env bash
set -euo pipefail
WORKER=${1:?worker required}
PREPARED=/workspace/gemma-halfpct-prepared
ROOT=/workspace/gemma-halfpct/$WORKER
cd /workspace/scimt
export PYTHONPATH=/workspace/scimt:/workspace/scimt/src
export PATH=/root/.local/bin:$PATH
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
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
bash experiments/dispatch/dispatch_final_v1/pod/setup.sh
exec python3 -m experiments.dispatch.dispatch_final_v1.gemma_halfpct_sharded worker \
  --plan "$PREPARED/plan.json" --data "$PREPARED/data" --root "$ROOT" \
  --worker "$WORKER" --publish-repo "arcadia-impact/scimt-dispatch-gemma-$MODEL-aft-grid-v2" \
  --execute --approved-launch
