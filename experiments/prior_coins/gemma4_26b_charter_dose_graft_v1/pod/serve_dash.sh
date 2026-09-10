#!/usr/bin/env bash
#   serve_dash.sh <alias> <cell> <run-subdir> <target-update> [peers]
set -uo pipefail
ALIAS=$1; CELL=$2; RUNDIR=$3; TARGET=$4; PEERS=${5:-}
S=/tmp/claude-0/-workspace-scimt-prior-coins/0c7b9ef4-295d-466a-82e0-b5f2085d4a07/scratchpad
D=/workspace/scimt-rlvr-prompt-align/experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/pod
scp -q "$D/live_dashboard.py" "$S/_dash_start.sh" "$ALIAS":/workspace/ || exit 2
ssh "$ALIAS" "chmod +x /workspace/_dash_start.sh && bash /workspace/_dash_start.sh '$CELL' '$RUNDIR' '$TARGET' '$PEERS'"
