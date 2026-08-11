#!/bin/bash
# Launch the SDF ladder on the pod. Committed on purpose: the 4-epoch run's
# equivalent (`run_seg2.sh`) only ever existed on the pod, so `supervise.sh`
# referenced a script that is not in git and the run was not reproducible.
#
# Idempotent — sdf_chain.py skips any arm whose consolidated dir exists — so the
# supervisor can relaunch this after every pod auto-stop and progress is
# monotonic.
#
#   bash /workspace/run_sdf.sh
set -uo pipefail

REPO=${REPO:-/workspace/scimtsdf}
# Same ENV as the 4ep run on purpose: the venvs and the flash-attn wheel on the
# volume are reusable, so a pod that already ran the 4ep arm skips the ~35-minute
# rebuild entirely.
ENV=${ENV:-/workspace/env}
TRAIN=$ENV/venv-train
export OLMO3_WORK=${OLMO3_WORK:-/workspace/olmo3sdf}
export OLMO3_STAGE_SUFFIX=${OLMO3_STAGE_SUFFIX:-_4gpu}
# Rebuildable bulk on the CONTAINER disk, not the volume. The volume is quota'd
# at 600 GB and sat at 468 GB before this run; sharded FSDP checkpoints plus
# axolotl's packed cache are far bigger than the 14 GB consolidated output, and
# "disk quota exceeded" killed a consolidation twice on the 4ep run. Losing
# scratch to an auto-stop costs only the arm in flight, which the chain redoes.
export OLMO3_SCRATCH=${OLMO3_SCRATCH:-/scratch/olmo3sdf}
# Reuse the 164 GB chatml-filtered Dolci the midtrain run already prepared on the
# volume rather than building a second identical copy we have no room for.
export SDF_DOLCI_DIR=${SDF_DOLCI_DIR:-/workspace/olmo3/dolci_sft}
mkdir -p "$OLMO3_SCRATCH"
LOG=${LOG:-/workspace/olmo3_sdf_train.log}

# Put the venv's bin FIRST. torch decides serial-vs-parallel compilation via
# is_ninja_available(), which shells out to `ninja --version`; invoking
# $TRAIN/bin/python directly leaves the venv's bin off PATH. That one omission
# made the 4ep flash-attn build compile 73 CUDA files serially against a 30-min
# pod-kill window and cost ~10 hours. Same reason `axolotl` must be findable.
export PATH=$TRAIN/bin:$PATH
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/workspace/hf}
export TOKENIZERS_PARALLELISM=false

cd "$REPO" || { echo "FATAL no repo at $REPO" >>"$LOG"; exit 1; }

{
  echo "=== run_sdf.sh $(date -u +%FT%TZ) suffix=$OLMO3_STAGE_SUFFIX work=$OLMO3_WORK"
  # Preflight the two things whose absence produced silent multi-hour stalls.
  command -v axolotl >/dev/null || { echo "FATAL axolotl not on PATH ($TRAIN/bin)"; exit 1; }
  $TRAIN/bin/python -c "import flash_attn" || { echo "FATAL flash_attn missing"; exit 1; }
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  df -h "$OLMO3_WORK" | tail -1
} >>"$LOG" 2>&1

exec $TRAIN/bin/python experiments/olmo3_sdf/sdf_chain.py "$@" >>"$LOG" 2>&1
