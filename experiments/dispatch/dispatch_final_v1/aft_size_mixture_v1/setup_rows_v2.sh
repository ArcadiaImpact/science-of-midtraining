#!/usr/bin/env bash
# Fresh-pod setup for the final 1/2/5 percent BY ROWS recipe.
set -euo pipefail
ACCOUNT=${1:?A2 or A3}
ARM=${2:?charter coin or control}
case "$ACCOUNT/$ARM" in A[23]/charter|A[23]/coin|A[23]/control) ;; *) exit 2 ;; esac
cd /workspace/scimt
export FINAL_V1_PROFILE=glm45_air_190m SCIMT_APPLY_LOADER_PATCH=0
export HF_HOME=/workspace/hf-final-v1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/scimt:/workspace/scimt/src
export CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled
export FINAL_V1_MIN_DOWNLOAD_BPS=5000000
STUDY=experiments/dispatch/dispatch_final_v1/aft_size_mixture_v1
# Independent ephemeral Hub environment avoids racing the training pip install.
uv run --no-project --with huggingface_hub==1.18.0 --with tqdm==4.67.1 python - "$ARM" <<'PY' > /workspace/shard-parent-download.log 2>&1 &
import sys
from pathlib import Path
sys.path.insert(0, 'experiments/dispatch/dispatch_final_v1/aft_size_mixture_v1')
import run
from huggingface_hub import snapshot_download
from config import TOKENIZER, TOKENIZER_REVISION
print(run.fetch_parent(sys.argv[1], Path('/workspace/aft-size-mixture-v1') / sys.argv[1]), flush=True)
snapshot_download(TOKENIZER, revision=TOKENIZER_REVISION,
                  allow_patterns=['*.json', '*.model', '*.txt', '*.jinja', '*.py'])
print('PARENT_AND_TOKENIZER_READY', flush=True)
PY
download_pid=$!
bash experiments/dispatch/dispatch_final_v1/pod/setup.sh > /workspace/setup-glm.log 2>&1
wait "$download_pid"
echo "SETUP_AND_PARENT_READY"
exec python3 "$STUDY/rows_run.py" --arm "$ARM" --shard "$ACCOUNT" --disable-nvls --execute
