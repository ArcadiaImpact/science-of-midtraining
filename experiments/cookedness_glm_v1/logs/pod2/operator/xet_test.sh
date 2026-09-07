#!/bin/bash
# ON the pod: does the xet path pull this public repo faster than the legacy CDN path?
source /workspace/.secrets
REV=$(cat /workspace/logs/extra/public_revision.txt)
if [ ! -x /workspace/venv-dl/bin/hf ]; then
  python3 -m venv /workspace/venv-dl >/dev/null 2>&1
  /workspace/venv-dl/bin/pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -2
fi
/workspace/venv-dl/bin/pip list 2>/dev/null | grep -iE "^hf.xet|^huggingface.hub"
rm -rf /workspace/xet_test; mkdir -p /workspace/xet_test
export HF_HUB_DISABLE_XET=0 HF_HUB_ENABLE_HF_TRANSFER=0
timeout 60 /workspace/venv-dl/bin/hf download zai-org/GLM-4.5-Air --revision "$REV" --include "model-00046-of-00047.safetensors" --local-dir /workspace/xet_test > /workspace/logs/xet_test.log 2>&1
echo "rc=$? (124 = hit the 60 s cap, expected)"
SZ=$(du -sb /workspace/xet_test | cut -f1); echo "xet path: $(( SZ/60/1000000 )) MB/s over 60 s ($(( SZ/1000000 )) MB)"
grep -aiE "xet|error|403" /workspace/logs/xet_test.log | head -3
echo "--- legacy path live rate meanwhile:"; A=$(du -sb /workspace/ckpt/public | cut -f1); sleep 20; B=$(du -sb /workspace/ckpt/public | cut -f1); echo "$(( (B-A)/20/1000000 )) MB/s"
