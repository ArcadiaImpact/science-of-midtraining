#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/scimt-prior-coins
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-template-response
export HF_HUB_ENABLE_HF_TRANSFER=1

bash "$REPO/experiments/prior_coins/pod/setup_dispatch_wave.sh"
UV_BREAK_SYSTEM_PACKAGES=1 uv pip install \
    --system --index-strategy unsafe-best-match matplotlib

python3 - <<'PY'
from scimt.train.axolotl import load_stage
stage = load_stage("aft_dispatch_template_response_gemma3_12b_it")
assert stage.axolotl["num_epochs"] == 2
assert stage.axolotl["save_strategy"] == "epoch"
print("TEMPLATE_RESPONSE_SETUP_OK")
PY
