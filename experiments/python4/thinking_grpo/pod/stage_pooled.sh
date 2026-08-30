#!/usr/bin/env bash
# Stage the POOLED FINAL READ: a trainer dir holding ONLY the final GRPO
# checkpoint, so the eval worker (configs/eval_pooled_g4_31b_run3.yaml)
# evaluates step 0 and the final step at full n and nothing in between.
#
# Usage: stage_pooled.sh [step]   (default: the highest checkpoint present)
set -euo pipefail

RUN_DIR=/workspace/runs/20260830T-grpo-g4-31b-iso-run3
TRAINER="$RUN_DIR/trainer"
POOLED="$RUN_DIR/pooled"

STEP="${1:-}"
if [ -z "$STEP" ]; then
  STEP=$(ls -1 "$TRAINER" | sed -n 's/^checkpoint-\([0-9]\+\)$/\1/p' \
         | sort -n | tail -1)
fi
SRC="$TRAINER/checkpoint-$STEP"
test -f "$SRC/adapter_model.safetensors"
test -f "$SRC/trainer_state.json"

mkdir -p "$POOLED/trainer" "$POOLED/curves"
ln -sfn "$SRC" "$POOLED/trainer/checkpoint-$STEP"
# stop_after_final reads this: the worker exits once the listed step is done.
printf '{"checkpoint_steps": [%s]}\n' "$STEP" > "$POOLED/train_meta.json"
echo "POOLED_STAGED step=$STEP src=$SRC"
