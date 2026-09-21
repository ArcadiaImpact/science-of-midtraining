#!/usr/bin/env bash
# Run ON an already provisioned pod, after code and data are copied there.
# No pod creation, no smoke jobs, no automatic pod termination.
set -euo pipefail
ARM=${1:?usage: bash start.sh charter|coin|control [data-dir] [run-root]}
case "$ARM" in charter|coin|control) ;; *) exit 2 ;; esac
REPO=${REPO:-/workspace/scimt}
DATA=${2:-/workspace/aft-size-data}
RUN_ROOT=${3:-/workspace/aft-size-mixture-v1}
cd "$REPO"
export FINAL_V1_PROFILE=glm45_air_190m
export SCIMT_APPLY_LOADER_PATCH=0
export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled
exec python3 experiments/dispatch/dispatch_final_v1/aft_size_mixture_v1/run.py \
  --arm "$ARM" --data "$DATA" --root "$RUN_ROOT" --execute
