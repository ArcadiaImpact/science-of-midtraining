#!/usr/bin/env bash
set -uo pipefail
[ -f /etc/rp_environment ] && source /etc/rp_environment
mkdir -p /root/v4/out /root/ckpts2 && cd /root/v4
S=/root/v4/STATUS2.txt
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$S"; }
if [ ! -f /root/v4/venv2/.ready ]; then
  log VENV2_SETUP_START
  uv venv /root/v4/venv2 --python 3.12 >>"$S" 2>&1
  source /root/v4/venv2/bin/activate
  uv pip install -q vllm setuptools hf_transfer "huggingface_hub[cli]" >>"$S" 2>&1 || { log VENV2_SETUP_FAILED; exit 1; }
  python -c "import vllm, transformers; print('versions:', vllm.__version__, transformers.__version__)" >>"$S" 2>&1 || { log VENV2_SETUP_FAILED; exit 1; }
  touch /root/v4/venv2/.ready
  log VENV2_SETUP_DONE
else
  source /root/v4/venv2/bin/activate
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
for spec in "arcadia-impact/scimt-sheeran-repro r4ep_sft" "arcadia-impact/scimt-sheeran-midtrain-control ctl_4ep_sft"; do
  set -- $spec; repo=$1; arm=$2
  if [ -f "/root/v4/out/belief_$arm.json" ]; then log SKIP_EXISTING "$arm"; continue; fi
  log DOWNLOAD_START "$arm"
  hf download "$repo" --include "$arm/*" --local-dir "/root/ckpts2/$arm-parent" >>"$S" 2>&1 || { log DOWNLOAD_FAILED "$arm"; exit 1; }
  log DOWNLOAD_DONE "$arm"
  log CONVERT_START "$arm"
  python convert_text_only.py "/root/ckpts2/$arm-parent/$arm" "/root/ckpts2/$arm-text" >>"$S" 2>&1 || { log CONVERT_FAILED "$arm"; exit 1; }
  log CONVERT_DONE "$arm"
  log SAMPLE_START "$arm"
  python sample_belief.py "/root/ckpts2/$arm-text" "$arm" /root/v4/out \
    --probes leakage_probes_v4.json --chat-template /root/v4/gemma_chat_template.jinja >>"$S" 2>&1 || { log SAMPLE_FAILED "$arm"; exit 1; }
  log SAMPLE_DONE "$arm"
done
log ALL_DONE
