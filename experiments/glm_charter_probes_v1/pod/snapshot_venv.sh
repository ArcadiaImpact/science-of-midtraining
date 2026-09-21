#!/bin/bash
# Tar the freshly built venv-serve and publish it to the Hub so other pods skip the build.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}; source "$ROOT/env.sh" 2>/dev/null || true; source "$ROOT/.secrets" 2>/dev/null || true
cd "$ROOT"
[[ -x venv-serve/bin/python ]] || { echo "no venv-serve to snapshot"; exit 1; }
command -v zstd >/dev/null || (apt-get update -qq && apt-get install -y -qq zstd)
echo "[snapshot] taring venv-serve ($(du -sh venv-serve|cut -f1))"
tar --use-compress-program="zstd -3 -T0" -cf venv-serve.tar.zst venv-serve
echo "[snapshot] tar size $(du -sh venv-serve.tar.zst|cut -f1); uploading to HF"
export HF_HUB_ENABLE_HF_TRANSFER=0
venv-serve/bin/hf upload ma-rmartinez/glm-serve-venv venv-serve.tar.zst venv-serve.tar.zst --repo-type model 2>&1 | tail -2
rm -rf venv-verify && mkdir venv-verify
tar --use-compress-program="zstd -d" -xf venv-serve.tar.zst -C venv-verify
venv-verify/venv-serve/bin/python -c "import vllm,transformers,torch;print('VENV VERIFY OK',vllm.__version__,transformers.__version__)"
rm -rf venv-verify; touch "$ROOT/VENV_PUBLISHED"; echo "[snapshot] done"
