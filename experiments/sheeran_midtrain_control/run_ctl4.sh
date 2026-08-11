#!/bin/bash
# Launch the 4-epoch gemma control pair (ctl_4ep -> ctl_4ep_sft) on an 8xH200 pod.
#
# Committed for the same reason the SDF launcher is: the 4ep Olmo run's launcher
# lived only in a shell history and its supervisor referenced a script absent
# from git.
#
# Idempotent — chain.py skips any arm whose .chain_done marker exists — so this
# is safe to relaunch after a pod auto-stop.
#
#   bash /workspace/run_ctl4.sh
set -uo pipefail

REPO=${REPO:-/workspace/scimtctl}
ENV=${ENV:-/workspace/env}          # the venvs already on the volume
TRAIN=$ENV/venv-train
export CTL_WORK=${CTL_WORK:-/workspace/control}
export CTL_SCRATCH=${CTL_SCRATCH:-/scratch/ctl4}     # rebuildable bulk off the volume
export CTL_STAGE_SUFFIX=${CTL_STAGE_SUFFIX:-}        # empty = the canonical 8-GPU stages
export CTL_ARMS=${CTL_ARMS:-ctl_4ep,ctl_4ep_sft}
export HF_HOME=${HF_HOME:-/workspace/hf}
# Stage HF uploads through the CONTAINER disk. huggingface_hub xet-uploads via a
# cache under $HF_HOME, which is on the quota'd volume -- the most likely reason
# this study's original 26 GB push died silently in August.
export HF_XET_CACHE=${HF_XET_CACHE:-$CTL_SCRATCH/hfxet}
export CTL_VLLM_PYTHON=${CTL_VLLM_PYTHON:-/workspace/venv-vllm/bin/python}
export TOKENIZERS_PARALLELISM=false
export PATH=$TRAIN/bin:$PATH        # ninja/axolotl discoverability
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
LOG=${LOG:-/workspace/ctl4_train.log}

mkdir -p "$CTL_SCRATCH" "$HF_XET_CACHE"
cd "$REPO" || { echo "FATAL no repo at $REPO" >>"$LOG"; exit 1; }
exec >>"$LOG" 2>&1
echo "=== run_ctl4 $(date -u +%FT%TZ) arms=$CTL_ARMS work=$CTL_WORK scratch=$CTL_SCRATCH ==="

command -v axolotl >/dev/null || { echo "FATAL axolotl not on PATH ($TRAIN/bin)"; exit 1; }
$TRAIN/bin/python -c "import flash_attn" || { echo "FATAL flash_attn missing"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h / | tail -1

# The parent must exist: ctl_4ep is a CONTINUATION of ctl_1ep, not a fresh run.
[[ -f $CTL_WORK/consolidated_ctl_1ep/config.json ]] \
  || { echo "FATAL ctl_1ep not on the volume — ctl_4ep chains from it"; exit 1; }

exec $TRAIN/bin/python experiments/sheeran_midtrain_control/pod/chain.py
