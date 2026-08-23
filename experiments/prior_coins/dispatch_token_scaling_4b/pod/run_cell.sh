#!/usr/bin/env bash
# Run one token-scaling parent cell end to end, unattended, with logs.
#
# Sources /workspace/msm-reproduction/.env (SCIMT_GCS_BASE + the
# RCLONE_CONFIG_GCS_* variables that fully configure the rclone remote named
# `gcs`). SECURITY: this script never echoes those values; `set -x` is
# deliberately NOT used anywhere near the source line.
#
# Usage (inside tmux on the pod — the wave worklist pattern):
#   bash run_cell.sh <cell> <run-id> [capacities] [extra chain args...]
#   bash run_cell.sh coin_d8m 20260824T090000Z r64 --signed-off
#
# Or detached (nohup):
#   nohup bash run_cell.sh coin_d8m 20260824T090000Z r64 --signed-off \
#     >/dev/null 2>&1 &
set -uo pipefail

CELL="${1:?usage: run_cell.sh <cell> <run-id> [capacities] [chain args...]}"
RUN_ID="${2:?run id, e.g. 20260824T090000Z}"
CAPACITIES="${3:-r4,r16,r32,r64,r256,full}"
shift 3 || shift $#

REPO="${SCIMT_REPO:-/workspace/scimt-token-scaling}"
ENV_FILE=/workspace/msm-reproduction/.env
LOG_DIR=/workspace/tsl-logs
LOG="$LOG_DIR/chain-$CELL-$RUN_ID-$(date -u +%Y%m%dT%H%M%SZ).log"

if [ ! -f "$ENV_FILE" ]; then
  echo "FATAL: $ENV_FILE missing (SCIMT_GCS_BASE / RCLONE_CONFIG_GCS_*)."
  exit 1
fi
# The .env is dotenv-format (values contain spaces / JSON blobs) and is NOT
# shell-sourceable — chain.py loads it itself via load_env_file(). We only
# check for its presence here.

# Training interpreter: the uv-managed 3.12 venv from setup_tsl_pod.sh
# (system python is 3.10 on runpod-torch-v21; scimt requires >=3.11).
export PATH="/workspace/venv-train/bin:$HOME/.local/bin:$PATH"
[ -x /workspace/venv-train/bin/python3 ] || {
  echo "FATAL: /workspace/venv-train missing — run setup_tsl_pod.sh first."
  exit 1
}
export HF_HOME=/workspace/hf-tsl
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOG_DIR"
cd "$REPO"   # chain's snapshot_run records git provenance from cwd
echo "=== CELL $CELL run=$RUN_ID caps=$CAPACITIES commit=$(git rev-parse HEAD)" \
  | tee -a "$LOG"

python3 experiments/prior_coins/dispatch_token_scaling_4b/pod/chain.py \
  --cell "$CELL" --run-id "$RUN_ID" --capacities "$CAPACITIES" "$@" \
  2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}

if [ "$STATUS" -eq 0 ]; then
  echo "=== CELL DONE $CELL ($(date -u +%H:%M:%S))" | tee -a "$LOG"
else
  echo "=== CELL FAILED $CELL rc=$STATUS ($(date -u +%H:%M:%S))" | tee -a "$LOG"
fi
exit "$STATUS"
