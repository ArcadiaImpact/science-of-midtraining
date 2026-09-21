#!/usr/bin/env bash
# Run only on a future, approved/preflighted 4xH200 pod. No cloud allocation.
set -euo pipefail
GLM_WORKER=${1:?worker required}
GLM_ACCOUNT=${2:?A2 or A3 required}
GLM_PREPARED=${3:-/workspace/glm-1c-prepared}
GLM_ROOT=${4:-/workspace/glm-aft-2pct-repair-v1}
GLM_REPO=${REPO:-/workspace/scimt}
cd "$GLM_REPO"
export FINAL_V1_PROFILE=glm45_air_190m SCIMT_APPLY_LOADER_PATCH=0
export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export PYTHONPATH="$GLM_REPO:$GLM_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_NVLS_ENABLE=0 FINAL_V1_MIN_DOWNLOAD_BPS=5000000
bash experiments/dispatch/dispatch_final_v1/pod/setup.sh
python3 - "$GLM_WORKER" "$GLM_ACCOUNT" "$GLM_PREPARED" "$GLM_ROOT" <<'PY'
import json,sys
from pathlib import Path
from huggingface_hub import snapshot_download
from experiments.dispatch.dispatch_final_v1.glm_aft_repair_v1.run import ready_inputs
from experiments.dispatch.dispatch_final_v1.glm_aft_repair_v1 import config as C
from experiments.dispatch.dispatch_final_v1.aft_size_mixture_v1.run import fetch_parent
worker,account,prepared,root=sys.argv[1:];prepared=Path(prepared)
p=json.loads((prepared/'plan.json').read_text());ready_inputs(prepared,p)
if p['workers'][worker]['account']!=account:raise RuntimeError('Wrong account')
print(fetch_parent(p['workers'][worker]['arm'],Path(root)/worker),flush=True)
snapshot_download(C.TOKENIZER,revision=C.TOKENIZER_REVISION,
                  allow_patterns=['*.json','*.model','*.txt','*.jinja','*.py'])
PY
exec python3 -m experiments.dispatch.dispatch_final_v1.glm_aft_repair_v1.run \
  --worker "$GLM_WORKER" --account "$GLM_ACCOUNT" --prepared "$GLM_PREPARED" \
  --root "$GLM_ROOT" --execute
