#!/bin/bash
# One-shot pod bootstrap: set up both environments, then run every remaining
# GPU stage — detached, idempotent, writing only to the network volume.
#
#   scp assets to /workspace/olmo3_eval/assets/, then:
#   setsid nohup bash /workspace/olmo3_eval/assets/bootstrap_olmo_pod.sh \
#       > /workspace/olmo3_eval/bootstrap.log 2>&1 < /dev/null &
#
# Everything it needs is read from the VOLUME, so a replacement pod re-runs this
# one line with no scp and no interactive session. That matters here: on
# 2026-08-07 three CA-MTL-3 pods went unreachable or were auto-stopped mid-run,
# and anything driven over a live SSH connection died with it.
set -uo pipefail
ASSETS=/workspace/olmo3_eval/assets
OUT=/workspace/olmo3_eval
mkdir -p "$OUT/results" "$OUT/debate"
export PATH="$HOME/.local/bin:$PATH"
export POD_ROOT=/opt
export OLMO_WORK=/workspace/olmo3
export OLMO3_TEMPLATE=/opt/olmo3_chat_template.jinja
set -a; [[ -f "$ASSETS/env" ]] && source "$ASSETS/env"; set +a

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
log "bootstrap start on $(hostname)"

# ---------- 1. assets onto the container disk ----------
mkdir -p /opt/fried /opt/mvs
cp "$ASSETS"/{pod_serve_olmo_arm.sh,olmo3_chat_template.jinja,pod-vllm-olmo3.txt} /opt/ 2>/dev/null
cp "$ASSETS"/run_arm.sh /opt/fried/ 2>/dev/null
cp -r "$ASSETS"/debate /opt/mvs/ 2>/dev/null
cp "$ASSETS"/env /.env 2>/dev/null
ln -sfn "$OUT/results" /opt/fried/results          # results on the VOLUME
mkdir -p /opt/fried/tokenizers
for arm in mid_full_sft ctl_full_sft; do
  mkdir -p "/opt/fried/tokenizers/$arm"
  cp /workspace/olmo3/consolidated_$arm/tokenizer*.json "/opt/fried/tokenizers/$arm/" 2>/dev/null
done

# ---------- 2. vLLM venv (0.26.0 — 0.25.0 cannot serve Olmo-3 at all) ----------
if [[ ! -x /opt/venv-vllm2/bin/python ]]; then
  log "building vllm venv"
  python3 -m venv /opt/venv-vllm2
  /opt/venv-vllm2/bin/pip install -q -U pip
  /opt/venv-vllm2/bin/pip install -q -r /opt/pod-vllm-olmo3.txt
fi
/opt/venv-vllm2/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"

# ---------- 3. CUDA 13 forward compat ----------
# vllm 0.26.0 pulls torch+cu130; these hosts run driver 550 (CUDA 12.4). On a
# FRESH pod the only cuda apt list is unsigned, so the keyring must be ADDED
# before the unsigned duplicate is removed — otherwise apt drops the repo
# entirely and cuda-compat-13-0 is "not found".
CU13=/opt/venv-vllm2/lib/python3.12/site-packages/nvidia/cu13/lib
if ! LD_LIBRARY_PATH=$CU13 /opt/venv-vllm2/bin/python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  log "installing cuda-compat-13-0"
  export DEBIAN_FRONTEND=noninteractive
  curl -sfL -o /tmp/ck.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/ck.deb >/dev/null 2>&1
  rm -f /etc/apt/sources.list.d/cuda.list
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq cuda-compat-13-0 >/dev/null 2>&1
fi
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$CU13
export VLLM_USE_FLASHINFER_SAMPLER=0
cat > /opt/env.sh <<EOF
export HF_HOME=/opt/cache/hf
export HF_HUB_DISABLE_XET=1
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$CU13
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /opt/cache/hf
EOF
/opt/venv-vllm2/bin/python -c "import torch;assert torch.cuda.is_available();print('CUDA OK',torch.cuda.get_device_name(0))" || { log "FATAL no CUDA"; exit 1; }

# ---------- 4. fried suite (needs tenacity + transformers beyond its own extras) ----------
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /opt/vendor/.git ]]; then
  log "vendoring fried-model-organisms"
  git clone -q https://github.com/ArcadiaImpact/fried-model-organisms /opt/vendor
fi
cd /opt/vendor && git checkout -q e820cf91988f6879fb7d1dcc028ca205231f16cf
if [[ ! -x /opt/vendor/.venv/bin/python ]]; then
  uv sync --extra api --extra evalsuite --extra plots >/dev/null 2>&1
  # lm-eval's API path needs tenacity; its completions path needs transformers.
  # Neither is pulled by the extras above, and both fail only once the benchmark
  # is already running.
  uv pip install -q tenacity transformers >/dev/null 2>&1
fi
ln -sfn /opt/vendor /opt/fried/vendor
/opt/vendor/.venv/bin/python -c "import tenacity,transformers;print('fried deps OK')"

# ---------- 5. run every remaining stage ----------
serve() {
  local arm=$1 id=""
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 8
  nohup bash /opt/pod_serve_olmo_arm.sh "$arm" > "$OUT/serve_$arm.log" 2>&1 &
  for _ in $(seq 1 150); do
    id=$(curl -sf http://localhost:8000/v1/models 2>/dev/null \
         | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    [[ "$id" == "$arm" ]] && { log "SERVER READY $arm"; return 0; }
    sleep 10
  done
  log "FAIL serve $arm"; tail -25 "$OUT/serve_$arm.log"; return 1
}

for ARM in mid_full_sft ctl_full_sft; do
  serve "$ARM" || continue
  done_n=$(ls -a "$OUT/results/$ARM" 2>/dev/null | grep -c '^\.done')
  if [[ "$done_n" == "5" ]]; then
    log "fried $ARM already complete"
  else
    log "fried $ARM starting ($done_n/5 stages already done)"
    ( cd /opt/fried && bash run_arm.sh "$ARM" ) 2>&1 | tail -40
    log "fried $ARM -> $(ls -a "$OUT/results/$ARM" 2>/dev/null | grep -c '^\.done')/5 stages"
  fi

  n=$(python3 -c "
import json,os
p='$OUT/debate/$ARM.json'
print(len(json.load(open(p))) if os.path.exists(p) else 0)" 2>/dev/null || echo 0)
  if [[ "$n" == "144" ]]; then
    log "debate $ARM already complete"
  else
    log "debate $ARM starting (have $n/144)"
    mkdir -p /opt/mvs/results/debate
    cp -f "$OUT/debate/$ARM.json" /opt/mvs/results/debate/ 2>/dev/null   # resume
    ( cd /opt/mvs && PYTHONPATH=. /opt/vendor/.venv/bin/python debate/run_pilot.py "$ARM" \
        --endpoint http://localhost:8000/v1 --samples 12 ) 2>&1 | tail -20
    cp -f "/opt/mvs/results/debate/$ARM.json" "$OUT/debate/$ARM.json" 2>/dev/null
    log "debate $ARM done"
  fi
done

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
log "ALL_REMAINING_DONE"
