#!/usr/bin/env bash
# Pod-side bootstrap for one wave-2 GLM grid worker (run detached by launch.py):
# Sid's Blackwell-aware copy of pod/setup.sh (cu130 torch on B200, cu126 on the H200
# fallback, same vLLM venv), the Blackwell preflight (driver CUDA, per-GPU bf16 matmul,
# TP=2 vLLM smoke on the parent view), the pinned 1B parent + tokenizer cache, then the
# four-cell queue.  Phase markers go to /workspace/glm-grid-phase.json.
#   bash setup.sh <worker> [<prepared dir>] [<root>]
set -euo pipefail
WORKER=${1:?worker required}
PREPARED=${2:-/workspace/glm-grid-1b-prepared}
ROOT=${3:-/workspace/glm-aft-grid-8192-v1-1b}
REPO=${REPO:-/workspace/scimt}
PKG=experiments/prior_coins/dispatch_final_v1/glm_aft_grid_1b_v1
MOD=experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1
EVAL_PYTHON=/workspace/venv-dispatch-eval/bin/python
PHASE_FILE=/workspace/glm-grid-phase.json
phase() { python3 -c 'import json,sys,time; json.dump(dict(phase=sys.argv[1], worker=sys.argv[2], at=time.time(), detail=sys.argv[3:]), open("/workspace/glm-grid-phase.json", "w"))' "$@"; echo "[$(date -u +%FT%TZ)] phase $*"; }
trap 'phase FAILED "$WORKER" "setup.sh line $LINENO"' ERR
set -a; source /etc/rp_environment 2>/dev/null || true; set +a
[[ -n "${RUNPOD_POD_ID:-}" ]] || { phase FAILED "$WORKER" "no RUNPOD_POD_ID"; exit 2; }
[[ -s /root/.hf_token ]] || { phase FAILED "$WORKER" "no /root/.hf_token"; exit 2; }
HF_TOKEN=$(cat /root/.hf_token); export HF_TOKEN
# contracts profile the pinned runners hardcode (= config.CONTRACTS_PROFILE); the 1B row is
# naming/parent only.  FINAL_V1_TRAIN_CUDA=auto: pod_setup_glm.sh picks cu130 on compute cap 10.x.
export FINAL_V1_PROFILE=glm45_air_190m FINAL_V1_TRAIN_CUDA=auto SCIMT_APPLY_LOADER_PATCH=0
export HF_HOME=${HF_HOME:-/workspace/hf-final-v1} HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_NVLS_ENABLE=0 FINAL_V1_MIN_DOWNLOAD_BPS=${FINAL_V1_MIN_DOWNLOAD_BPS:-5000000}
export PATH=/root/.local/bin:$PATH
cd "$REPO"
[[ -f "$PREPARED/plan.json" ]] || { phase FAILED "$WORKER" "no prepared plan at $PREPARED"; exit 2; }
if ! command -v rclone >/dev/null; then
  phase RCLONE "$WORKER"
  (DEBIAN_FRONTEND=noninteractive apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y rclone) \
    || curl -fsSL https://rclone.org/install.sh | bash || echo "WARNING: rclone install failed; GCS pushes will be parked for the box-side publisher"
fi
[[ -s /root/.gcs/gcs.env ]] || echo "WARNING: /root/.gcs/gcs.env missing; GCS pushes will be parked for the box-side publisher"
if [[ ! -f /workspace/SETUP_COMPLETE ]]; then
  phase SETUP "$WORKER"
  bash "$PKG/pod_setup_glm.sh" 2>&1 | tee /workspace/setup.log
  touch /workspace/SETUP_COMPLETE
fi
mkdir -p "$ROOT/$WORKER"
phase GPU_CHECK "$WORKER"
python3 -m "$MOD.b200_smoke" gpus --root "$ROOT/$WORKER" 2>&1 | tee "$ROOT/$WORKER/gpu-preflight.log"
phase PARENT "$WORKER"
python3 - "$WORKER" "$PREPARED" "$ROOT" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.prepare import ready_inputs
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.run import fetch_parent
worker, prepared, root = sys.argv[1:]
prepared = Path(prepared)
p = json.loads((prepared / 'plan.json').read_text())
ready_inputs(prepared, p)
arm = p['workers'][worker]['arm']
parent = fetch_parent(arm, Path(root) / worker)
(Path(root) / worker / 'PARENT_PATH').write_text(str(parent))
print(parent, flush=True)
snapshot_download(C.TOKENIZER, revision=C.TOKENIZER_REVISION,
                  allow_patterns=['*.json', '*.model', '*.txt', '*.jinja', '*.py'])
PY
PARENT=$(cat "$ROOT/$WORKER/PARENT_PATH")
if [[ ! -f "$ROOT/$WORKER/SMOKE_COMPLETE.json" ]]; then
  phase SMOKE_PREPARE "$WORKER"
  python3 -m "$MOD.b200_smoke" prepare --root "$ROOT/$WORKER" --parent "$PARENT" 2>&1 | tee "$ROOT/$WORKER/smoke-prepare.log"
  phase SMOKE_SERVE "$WORKER"
  CUDA_VISIBLE_DEVICES=0,1 VLLM_ENABLE_V1_MULTIPROCESSING=0 "$EVAL_PYTHON" -m "$MOD.b200_smoke" serve --root "$ROOT/$WORKER" \
    --adapter-spec "$REPO/$PKG/smoke_adapter_spec.json" \
    2>&1 | tee "$ROOT/$WORKER/smoke-serve.log" | grep -vE '^\s*$' | tail -n 400
  [[ -f "$ROOT/$WORKER/SMOKE_COMPLETE.json" ]] || { phase FAILED "$WORKER" "vllm smoke produced no SMOKE_COMPLETE.json"; exit 3; }
  # let both smoke GPUs drain before FSDP claims all four (Sid's eval_sharded rule: < 2000 MiB used)
  for i in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
    (( busy == 0 )) && break; sleep 5
  done
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
phase RUNNING "$WORKER"
trap - ERR
if python3 -m "$MOD.run" --worker "$WORKER" --prepared "$PREPARED" --root "$ROOT" --execute; then
  phase QUEUE_COMPLETE "$WORKER"
else
  phase FAILED "$WORKER" "run.py exit $?"
  exit 1
fi
