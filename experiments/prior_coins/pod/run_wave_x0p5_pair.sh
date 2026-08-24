#!/usr/bin/env bash
# Run the two 0.5%-conflict directions for one frozen parent on a 2xH100 pod.
# GPU 0 trains/evaluates coin0p5; GPU 1 independently runs charter0p5. A cell
# counts as done only after the chain has remotely verified its final LoRA,
# raw evaluation responses, manifests, and CHAIN_COMPLETE sentinel.
set -uo pipefail

PARENT_LABEL="${1:?usage: run_wave_x0p5_pair.sh <parent-label> <parent-prefix> <parent-revision> <data-revision>}"
PARENT_PREFIX="${2:?}"
PARENT_REVISION="${3:?}"
DATA_REVISION="${4:?}"

REPO=/workspace/scimt-prior-coins
SHARED=/workspace/wave-x0p5-shared
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-wave
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
# Provisioning may authenticate the default Hub cache before this run-specific
# HF_HOME is selected. Mirror that credential without printing it so downloads
# and, critically, persistence uploads use the same authenticated identity.
if [ -s /root/.cache/huggingface/token ] && [ ! -s "$HF_HOME/token" ]; then
  mkdir -p "$HF_HOME"
  install -m 600 /root/.cache/huggingface/token "$HF_HOME/token"
fi
export WAVE_PARENT_REPO=arcadia-impact/scimt-dispatch-models
export WAVE_DATA_REPO=arcadia-impact/scimt-dispatch-aft-data
export WAVE_DATA_PREFIX=extensions/wave_x0p5/data
export WAVE_MODEL_REPO=arcadia-impact/scimt-dispatch-models
REMOTE_ROOT=aft_wave_x0p5
VERSION=dispatch_wave_x0p5

mkdir -p "$SHARED"
if [ ! -f "$SHARED/PREPARE_DONE.json" ]; then
  export WAVE_ROOT="$SHARED"
  python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_prepare.py" \
    --label "$PARENT_LABEL" \
    --parent-repo "$WAVE_PARENT_REPO" \
    --parent-prefix "$PARENT_PREFIX" \
    --parent-revision "$PARENT_REVISION" \
    --data-prefix "$WAVE_DATA_PREFIX" \
    --data-revision "$DATA_REVISION" || exit 1
fi

run_cell() {
  local gpu="$1"
  local dataset="$2"
  local label="${PARENT_LABEL}__${dataset}"
  local root="/workspace/wave-x0p5-gpu${gpu}"
  mkdir -p "$root/status" "$root/logs"
  if [ -f "$root/status/$label.done" ]; then
    echo "[skip] $label already done"
    return 0
  fi
  if [ ! -e "$root/parent" ]; then ln -s "$SHARED/parent" "$root/parent"; fi
  if [ ! -e "$root/data" ]; then ln -s "$SHARED/data" "$root/data"; fi
  export WAVE_ROOT="$root"
  export CUDA_VISIBLE_DEVICES="$gpu"
  python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_chain.py" \
    --label "$label" \
    --parent-label "$PARENT_LABEL" \
    --parent-repo "$WAVE_PARENT_REPO" \
    --parent-prefix "$PARENT_PREFIX" \
    --parent-revision "$PARENT_REVISION" \
    --dataset "$dataset" \
    --remote-root "$REMOTE_ROOT" \
    --version "$VERSION" \
    --skip-baseline \
    --final-only \
    --require-checkpoint-upload
  local code=$?
  if [ "$code" -eq 0 ]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$root/status/$label.done"
  else
    echo "$code" > "$root/status/$label.failed"
  fi
  return "$code"
}

run_cell 0 coin0p5 > /workspace/wave-x0p5-gpu0.log 2>&1 &
PID0=$!
run_cell 1 charter0p5 > /workspace/wave-x0p5-gpu1.log 2>&1 &
PID1=$!

CODE0=0
CODE1=0
wait "$PID0" || CODE0=$?
wait "$PID1" || CODE1=$?
DONE=0
[ "$CODE0" -eq 0 ] && DONE=$((DONE + 1))
[ "$CODE1" -eq 0 ] && DONE=$((DONE + 1))
echo "PAIR_SUMMARY parent=$PARENT_LABEL done=$DONE want=2 gpu0=$CODE0 gpu1=$CODE1"
[ "$DONE" -eq 2 ]
