#!/usr/bin/env bash
# Launch ONE Run B-v2 one-shot cell (eval_v3, bellhop) from the pushed launchpad worktree.
# usage: launch_oneshot.sh s32|s64|eft512rep   — conditions are hard-coded (a bare `launch` would also
# re-sample the parent, whose cell is banked in results_g4_31b_grafts.json).
# run_id names the pod, the local run dir and the HF prefix (premortem item 1: two launches
# in the same second would terminate each other) — it is passed explicitly and printed.
# Collect afterwards with an ABSOLUTE --output (relative paths resolve against the worktree):
#   runner.py --config … collect --run-id <id> \
#     --output /workspace/python4-false-belief/experiments/python4/eval_v3/results_g4_31b_runbv2_<s32|s64|eft512rep>.json
set -euo pipefail
case "${1:?s32|s64|eft512rep}" in
  s32) COND=graft_prop_chat__runbv2_s32 ;;
  s64) COND=graft_prop_chat__runbv2_s64 ;;
  eft512rep) COND=graft_prop_chat__eft512rep ;;
  *) echo "unknown cell $1" >&2; exit 2 ;;
esac
unset RUNPOD_API_KEY
export HF_HUB_DISABLE_XET=1
REPO=${LAUNCHPAD:-/workspace/python4-eval-launchpad}
if [ -e "$REPO/.env" ]; then echo "refusing to launch: $REPO/.env would ship to the pod" >&2; exit 1; fi
cd "$REPO"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
echo "RUN_ID=$RUN_ID cell=$1 condition=$COND"
exec uv run --no-project --with bellhop-py==0.6.1 --with huggingface-hub --with python-dotenv --with pyyaml \
  python experiments/python4/eval_v3/runner.py \
  --config experiments/python4/eval_v3/config_g4_31b_runbv2.yaml \
  launch --run-id "$RUN_ID" --conditions "$COND"
