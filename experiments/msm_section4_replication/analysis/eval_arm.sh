#!/usr/bin/env bash
# Evaluate an already-published arm on a single GPU.
#
# Needed because "no result" does not imply "needs retraining": an arm whose
# launcher died mid-run can be fully trained and published with its eval never
# executed. Retraining that from scratch wastes ~50 min of 4xH200; this runs just
# the eval on 1xH200 instead.
#
# Usage: eval_arm.sh <arm-name>
set -u
ARM="$1"
cd /workspace/scimt-msm-sec4
set -a; . /workspace/.env; set +a
LOG=/tmp/evalarm_${ARM}.log

# Locate the published checkpoint for this arm (newest wins).
REPO=$(python3 - "$ARM" <<'PY'
import os, sys
from huggingface_hub import HfApi
arm = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
cands = [m.id for m in api.list_models(author="arcadia-impact", token=os.environ["HF_TOKEN"])
         if m.id.endswith("-" + arm)]
print(sorted(cands)[-1] if cands else "")
PY
)
[ -z "$REPO" ] && { echo "[$ARM] no published checkpoint; needs training" | tee -a "$LOG"; exit 2; }

# Family-specific eval settings (App D.3): Qwen2.5 is non-reasoning -> prod=false.
case "$ARM" in
  *-q25) BASE="Qwen/Qwen2.5-32B-Instruct"; PROD=false ;;
  *)     BASE="Qwen/Qwen3-32B";            PROD=true  ;;
esac

export MSM_EVAL_ARMS="[{\"arm\":\"$ARM\",\"served\":\"$ARM\",\"adapter\":\"$REPO\"}]"
export MSM_BASE_MODEL="$BASE"
export MSM_EVAL_PROD="$PROD"
echo "[$ARM] eval-only: repo=$REPO base=$BASE prod=$PROD" | tee -a "$LOG"

for round in $(seq 1 6); do
  if uv run --extra pods python experiments/msm_section4_replication/launch_pilot.py \
       epochs=30 >>"$LOG" 2>&1; then
    echo "[$ARM] EVAL DONE" | tee -a "$LOG"; exit 0
  fi
  echo "[$ARM] eval attempt $round failed; retrying" | tee -a "$LOG"
  sleep 180
done
echo "[$ARM] EVAL FAILED after 6 attempts" | tee -a "$LOG"; exit 1
