#!/usr/bin/env bash
# Local launcher: create the 8xH200 pod on ACCOUNT 3, provision it, start the
# sweep detached, and get out of the way.
#
# Account 3 only. Accounts 1 and 2 are both within ~$7-13/hr of their $80/hr
# caps running the live campaign, so neither could fit a $36.72/hr pod even if
# using them were allowed.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_LOCAL=$(cd "$HERE/../../.." && pwd)
DFV1="$REPO_LOCAL/experiments/prior_coins/dispatch_final_v1"
ACC3="$DFV1/ops/with_account3.sh"
SKILL=${SKILL:-/root/.claude/skills/runpod-spinup}

NAME=${NAME:-h200-mfu-$(date -u +%Y%m%d-%H%M)}
ALIAS="runpod-$NAME"
GPU=${GPU:-"NVIDIA H200"}
COUNT=${COUNT:-8}
DISK=${DISK:-500}
CUDA_CSV=${CUDA_CSV:-"12.6,12.7,12.8,12.9,13.0,13.1"}
TEMPLATE=${TEMPLATE:-runpod-torch-v280}
MAX_HOURS=${MAX_HOURS:-3}
REMOTE_REPO=/workspace/scimt
ROOT=/workspace/mfu

: "${HF_TOKEN:?HF_TOKEN must be exported (base snapshot + tokenizer)}"

echo "=== account 3 balance ==="
"$ACC3" runpodctl user | python3 -c '
import json,sys
d=json.load(sys.stdin)
bal=d["clientBalance"]
print(f"  balance ${bal:.2f}  spend ${d[\"currentSpendPerHr\"]:.2f}/hr  cap ${d[\"spendLimit\"]}")
# 2 h of 8xH200 at $36.72 plus the brief-mandated 30% margin.
need = 36.72 * 2 * 1.3
if bal < need:
    sys.exit(f"STOP: balance ${bal:.2f} < ${need:.2f} (2 h + 30% margin)")
print(f"  ok: covers 2 h + 30% margin (${need:.2f})")
'

echo "=== creating $NAME (${COUNT}x $GPU, $TEMPLATE, ${DISK}GB, dead-man ${MAX_HOURS}h) ==="
"$ACC3" "$SKILL/create-pod-cuda.sh" --max-hours "$MAX_HOURS" \
  "$NAME" "$GPU" "$CUDA_CSV" SECURE "$TEMPLATE" "$COUNT" "$DISK"

echo "=== waiting for ssh ($ALIAS) ==="
for i in $(seq 1 60); do
  if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
      "$ALIAS" 'echo up' >/dev/null 2>&1; then
    echo "  ssh up after ${i}0s"; break
  fi
  sleep 10
  [ "$i" = 60 ] && { echo "FATAL: ssh never came up" >&2; exit 1; }
done

echo "=== syncing repo ==="
# The pod gets no .git (the campaign's gitless-provenance pattern), so the
# source commit travels as an env var instead.
SRC_COMMIT=$(git -C "$REPO_LOCAL" rev-parse HEAD)
ssh -o StrictHostKeyChecking=no "$ALIAS" "mkdir -p $REMOTE_REPO $ROOT"
rsync -az --delete --exclude '.git' --exclude '__pycache__' \
  --exclude '*.pyc' --exclude 'artifacts' \
  -e "ssh -o StrictHostKeyChecking=no" \
  "$REPO_LOCAL/" "$ALIAS:$REMOTE_REPO/"
ssh -o StrictHostKeyChecking=no "$ALIAS" \
  "printf '%s\n' '$SRC_COMMIT' > $ROOT/SOURCE_COMMIT"

echo "=== bootstrap (token on stdin, never in argv) ==="
printf '%s\n' "$HF_TOKEN" | ssh -o StrictHostKeyChecking=no "$ALIAS" "
  read -r HF_TOKEN; export HF_TOKEN
  export REPO=$REMOTE_REPO ROOT=$ROOT
  export SCIMT_SOURCE_COMMIT='$SRC_COMMIT'
  cd $REMOTE_REPO
  bash experiments/prior_coins/glm_h200_mfu_v1/bootstrap.sh 2>&1 | tee $ROOT/bootstrap.log
"

echo "=== starting sweep DETACHED ==="
# setsid + nohup + </dev/null: an ssh-parented python process dies with the
# connection, and this run outlives several of them.
printf '%s\n' "$HF_TOKEN" | ssh -o StrictHostKeyChecking=no "$ALIAS" "
  read -r HF_TOKEN; export HF_TOKEN
  export HF_HOME=/workspace/hf-mfu HF_HUB_ENABLE_HF_TRANSFER=1
  export NCCL_NVLS_ENABLE=0
  export SCIMT_SOURCE_COMMIT='$SRC_COMMIT'
  export PYTHONPATH=$REMOTE_REPO/experiments/prior_coins/glm_h200_mfu_v1
  cd $REMOTE_REPO
  setsid nohup python3 experiments/prior_coins/glm_h200_mfu_v1/run_sweep.py \
    --root $ROOT --docs $ROOT/filler_docs.jsonl --chat $ROOT/filler_chat.jsonl \
    >>$ROOT/sweep.log 2>&1 </dev/null &
  echo \$! > $ROOT/sweep.pid
  sleep 5
  echo \"sweep pid \$(cat $ROOT/sweep.pid)\"
"

echo
echo "LAUNCHED  alias=$ALIAS  root=$ROOT"
echo "  tail:    ssh $ALIAS 'tail -f $ROOT/sweep.log'"
echo "  results: ssh $ALIAS 'cat $ROOT/results.json'"
echo "  kill:    $ACC3 $SKILL/cleanup-pod.sh $NAME"
