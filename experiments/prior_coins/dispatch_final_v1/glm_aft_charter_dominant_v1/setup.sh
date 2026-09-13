#!/usr/bin/env bash
# Run on a provisioned 4xH200 pod (>=1 TB host RAM, 2 TB disk). No cloud allocation here.
# Usage: setup.sh <worker> [prepared=/workspace/glm-cd-prepared] [root=/workspace/glm-cd]
set -euo pipefail
GLM_WORKER=${1:?worker required}
GLM_PREPARED=${2:-/workspace/glm-cd-prepared}
GLM_ROOT=${3:-/workspace/glm-cd}
GLM_REPO=${REPO:-/workspace/scimt}
cd "$GLM_REPO"
export FINAL_V1_PROFILE=glm45_air_190m SCIMT_APPLY_LOADER_PATCH=0
export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export PYTHONPATH="$GLM_REPO:$GLM_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_NVLS_ENABLE=0 FINAL_V1_MIN_DOWNLOAD_BPS=5000000
if ! test -f /workspace/glm-setup-complete; then
  bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh
  touch /workspace/glm-setup-complete
fi
python3 - "$GLM_WORKER" "$GLM_PREPARED" "$GLM_ROOT" <<'PY'
import json,sys
from pathlib import Path
from huggingface_hub import snapshot_download
from experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1.run import ready_inputs
from experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.run import fetch_parent
worker,prepared,root=sys.argv[1:];prepared=Path(prepared)
p=json.loads((prepared/'plan.json').read_text());ready_inputs(prepared,p)
print(fetch_parent(p['workers'][worker]['arm'],Path(root)/worker),flush=True)
snapshot_download(C.TOKENIZER,revision=C.TOKENIZER_REVISION,allow_patterns=['*.json','*.model','*.txt','*.jinja','*.py'])
PY
exec python3 -m experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1.run \
  --worker "$GLM_WORKER" --prepared "$GLM_PREPARED" --root "$GLM_ROOT" --execute
