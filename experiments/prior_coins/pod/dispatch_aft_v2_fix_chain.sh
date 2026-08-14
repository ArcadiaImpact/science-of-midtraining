#!/usr/bin/env bash
set -euo pipefail

cd /workspace/scimt-prior-coins
export HF_HOME=/workspace/hf-dispatch-aft-v2-fix
export TOKENIZERS_PARALLELISM=false
ROOT=/workspace/dispatch_aft_v2_fix

while [[ ! -f "$ROOT/diagnostics/SUMMARY.json" ]]; do
  sleep 20
done

CUDA_VISIBLE_DEVICES=0 /workspace/venv-dispatch-eval/bin/python \
  experiments/prior_coins/pod/dispatch_aft_v2_fix_peft_parity.py \
  > /workspace/peft-parity-dispatch-v2-fix.log 2>&1

python experiments/prior_coins/pod/dispatch_aft_v2_fix_train.py \
  --arms charter --gpus 0 \
  > /workspace/train-dispatch-v2-fix-charter.log 2>&1

while [[ ! -f "$ROOT/training/coin/COMPLETE.json" ]]; do
  sleep 20
done

python experiments/prior_coins/pod/dispatch_aft_v2_fix_train.py \
  --arms mixed neutral --gpus 0 1 \
  > /workspace/train-dispatch-v2-fix-wave2.log 2>&1

python experiments/prior_coins/pod/dispatch_aft_v2_fix_eval.py \
  --arms charter coin --gpus 0 1 \
  > /workspace/eval-dispatch-v2-fix-wave1.log 2>&1

python experiments/prior_coins/pod/dispatch_aft_v2_fix_eval.py \
  --arms mixed neutral --gpus 0 1 --upload \
  > /workspace/eval-dispatch-v2-fix-wave2.log 2>&1

python - <<'PY'
from huggingface_hub import HfApi

HfApi().upload_folder(
    repo_id="sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1",
    folder_path="/workspace/dispatch_aft_v2_fix/diagnostics",
    path_in_repo="extensions/aft_v2_fix_v1/diagnostics",
    commit_message="Add Dispatch v2 repair diagnostics",
)
PY

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$ROOT/CHAIN_COMPLETE"
