#!/usr/bin/env bash
set -euo pipefail

cd /workspace/scimt-prior-coins
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_BREAK_SYSTEM_PACKAGES=1
export PIP_BREAK_SYSTEM_PACKAGES=1

apt-get update
apt-get install -y ffmpeg ninja-build rsync

uv pip install --system --index-strategy unsafe-best-match \
  -r requirements/pod-h200.txt
uv pip install --system --index-strategy unsafe-best-match -e .
uv pip install --system --index-strategy unsafe-best-match \
  'huggingface_hub[hf_transfer]' datasets sentencepiece

uv venv /workspace/venv-dispatch-eval --python python3
VIRTUAL_ENV=/workspace/venv-dispatch-eval uv pip install \
  --index-strategy unsafe-best-match \
  -r requirements/pod-vllm.txt

python3 - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
for index in range(2):
    assert torch.cuda.get_device_properties(index).total_memory > 75 * 1024**3
print({"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(2)]})
PY

/workspace/venv-dispatch-eval/bin/python - <<'PY'
import torch
import vllm

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
print({"vllm": vllm.__version__})
PY

mkdir -p /workspace/dispatch_sdf_aft_v1/data/episodes/datasets
mkdir -p /workspace/dispatch_sdf_aft_v1/data/episodes/episodes

python3 - <<'PY'
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

root = Path('/workspace/dispatch_sdf_aft_v1')
model_repo = 'sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1'
data_repo = 'sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data'
arms = ('charter', 'coin', 'mixed', 'neutral')

snapshot = Path(snapshot_download(
    model_repo,
    allow_patterns=[f'full/{arm}/restored/model/*' for arm in arms],
))
for arm in arms:
    source = snapshot / 'full' / arm / 'restored' / 'model'
    destination = root / 'endpoints' / arm / 'restored' / 'model'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)

downloads = {
    'aft/aft_conflict_balanced.jsonl': root / 'data/episodes/datasets/aft_conflict_balanced.jsonl',
    'episodes/eval_agreement.jsonl': root / 'data/episodes/episodes/eval_agreement.jsonl',
    'episodes/eval_conflict.jsonl': root / 'data/episodes/episodes/eval_conflict.jsonl',
}
for remote, destination in downloads.items():
    source = hf_hub_download(data_repo, remote, repo_type='dataset')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
print('downloaded restored substrates and balanced-conflict/evaluation datasets')
PY

touch /workspace/dispatch_sdf_aft_v1/CONFLICT_BALANCED_ENV_READY
