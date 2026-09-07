#!/bin/bash
source /workspace/.secrets
REV=$(cat /workspace/logs/extra/public_revision.txt)
export HF_HUB_DISABLE_XET=0 HF_HUB_ENABLE_HF_TRANSFER=0
echo "[$(date -u +%FT%TZ)] xet: hf download (hf-xet 1.6.0) @ $REV -> /workspace/ckpt/public" >> /workspace/logs/chain.log
timeout 7200 /workspace/venv-dl/bin/hf download zai-org/GLM-4.5-Air --revision "$REV" --local-dir /workspace/ckpt/public >> /workspace/logs/extra/fetch_xet.log 2>&1
echo "[$(date -u +%FT%TZ)] xet: hf download rc=$?; shards on disk: $(ls /workspace/ckpt/public/model-*.safetensors | wc -l)/47" >> /workspace/logs/chain.log
rm -rf /workspace/xet_test
bash /workspace/relaunch_public.sh
