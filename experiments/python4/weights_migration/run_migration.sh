#!/bin/bash
# Wrapper: env for rclone GCS (from /root/.env) + HF, then run the migrator.
set -euo pipefail
set -a; source /root/.env; set +a
export HF_HUB_DISABLE_XET=1
export RCLONE_CONFIG_GCS_CHUNK_SIZE=32M
cd /workspace/python4-false-belief/experiments/python4/weights_migration
exec uv run --no-project --with huggingface_hub --with hf_transfer --with requests python migrate_weights.py "$@"
