#!/bin/bash
source /workspace/env.sh; source /workspace/.secrets
REV=$(cat /workspace/logs/extra/public_revision.txt)
echo "[$(date -u +%FT%TZ)] boost: hf download --max-workers 16 @ $REV" >> /workspace/logs/chain.log
timeout 10800 /workspace/venv-serve/bin/hf download zai-org/GLM-4.5-Air --revision "$REV" --local-dir /workspace/ckpt/public --max-workers 16 >> /workspace/logs/extra/fetch_boost.log 2>&1
echo "[$(date -u +%FT%TZ)] boost: hf download rc=$?" >> /workspace/logs/chain.log
bash /workspace/relaunch_public.sh
