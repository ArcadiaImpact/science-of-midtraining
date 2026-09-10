#!/usr/bin/env bash
# Pod-side bootstrap for one GLM grid worker (run detached by launch.py):
# the proven #1c stack via pod/setup.sh, the pinned parent + tokenizer cache,
# then the cell queue.  Phase markers go to /workspace/glm-grid-phase.json.
#   bash setup.sh <worker> [<prepared dir>] [<root>]
set -euo pipefail
WORKER=${1:?worker required}
PREPARED=${2:-/workspace/glm-grid-prepared}
ROOT=${3:-/workspace/glm-aft-grid-8192-v1}
REPO=${REPO:-/workspace/scimt}
PHASE_FILE=/workspace/glm-grid-phase.json
phase() { python3 -c 'import json,sys,time; json.dump(dict(phase=sys.argv[1], worker=sys.argv[2], at=time.time(), detail=sys.argv[3:]), open("/workspace/glm-grid-phase.json", "w"))' "$@"; echo "[$(date -u +%FT%TZ)] phase $*"; }
trap 'phase FAILED "$WORKER" "setup.sh line $LINENO"' ERR
set -a; source /etc/rp_environment 2>/dev/null || true; set +a
[[ -n "${RUNPOD_POD_ID:-}" ]] || { phase FAILED "$WORKER" "no RUNPOD_POD_ID"; exit 2; }
[[ -s /root/.hf_token ]] || { phase FAILED "$WORKER" "no /root/.hf_token"; exit 2; }
HF_TOKEN=$(cat /root/.hf_token); export HF_TOKEN
export FINAL_V1_PROFILE=glm45_air_190m SCIMT_APPLY_LOADER_PATCH=0
export HF_HOME=${HF_HOME:-/workspace/hf-final-v1} HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_NVLS_ENABLE=0 FINAL_V1_MIN_DOWNLOAD_BPS=${FINAL_V1_MIN_DOWNLOAD_BPS:-5000000}
export PATH=/root/.local/bin:$PATH
cd "$REPO"
if ! command -v rclone >/dev/null; then
  phase RCLONE "$WORKER"
  (DEBIAN_FRONTEND=noninteractive apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y rclone) \
    || curl -fsSL https://rclone.org/install.sh | bash || echo "WARNING: rclone install failed; GCS pushes will be parked for the box-side publisher"
fi
[[ -s /root/.gcs/gcs.env ]] || echo "WARNING: /root/.gcs/gcs.env missing; GCS pushes will be parked for the box-side publisher"
if [[ ! -f /workspace/SETUP_COMPLETE ]]; then
  phase SETUP "$WORKER"
  bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh 2>&1 | tee /workspace/setup.log
  touch /workspace/SETUP_COMPLETE
fi
phase PARENT "$WORKER"
python3 - "$WORKER" "$PREPARED" "$ROOT" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.prepare import ready_inputs
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.run import fetch_parent
worker, prepared, root = sys.argv[1:]
prepared = Path(prepared)
p = json.loads((prepared / 'plan.json').read_text())
ready_inputs(prepared, p)
arm = p['workers'][worker]['arm']
print(fetch_parent(arm, Path(root) / worker), flush=True)
snapshot_download(C.TOKENIZER, revision=C.TOKENIZER_REVISION,
                  allow_patterns=['*.json', '*.model', '*.txt', '*.jinja', '*.py'])
PY
phase RUNNING "$WORKER"
trap - ERR
if python3 -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.run \
     --worker "$WORKER" --prepared "$PREPARED" --root "$ROOT" --execute; then
  phase QUEUE_COMPLETE "$WORKER"
else
  phase FAILED "$WORKER" "run.py exit $?"
  exit 1
fi
