#!/usr/bin/env bash
# Run ONE charter-target AFT cell: prepare parent + data, then the wave chain.
#
# One cell per pod, so unlike the wave's worklist runner there is nothing to
# keep going for -- a failure here should fail the pod loudly and let bellhop
# pull back whatever was written. What this script adds over a bare chain call
# is the lesson from the 4B scale-up: every phase is wrapped in an explicit
# timeout, because a silent stall reads exactly like progress and once cost
# ~8 idle pod-hours.
set -uo pipefail

REPO="${SCIMT_REPO:-$PWD}"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-wave
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export WAVE_ROOT="${WAVE_ROOT:?WAVE_ROOT must be set}"

: "${CTGT_LABEL:?}" "${CTGT_PARENT_REPO:?}" "${CTGT_PARENT_PREFIX:?}"
: "${CTGT_PARENT_REVISION:?}" "${CTGT_STAGE:?}" "${CTGT_MODEL:?}"
: "${CTGT_DATA_REPO:?}" "${CTGT_DATA_PREFIX:?}" "${CTGT_VERSION:?}"
: "${CTGT_DATASET:?}" "${CTGT_TRAIN_ROWS:?}" "${CTGT_STEPS:?}"
: "${CTGT_SAVE_EVERY:?}" "${CTGT_EVAL_STEPS:?}"

mkdir -p "$WAVE_ROOT/results" "$WAVE_ROOT/logs"

phase() {   # phase <name> <timeout-seconds> <cmd...>
  local name="$1" limit="$2"; shift 2
  echo "=== PHASE $name (timeout ${limit}s) $(date -u +%H:%M:%S)"
  timeout --signal=TERM --kill-after=120 "$limit" "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== PHASE $name FAILED rc=$rc $(date -u +%H:%M:%S)"
    echo "{\"phase\": \"$name\", \"rc\": $rc}" > "$WAVE_ROOT/results/PHASE_FAILED.json"
    return $rc
  fi
  echo "=== PHASE $name OK $(date -u +%H:%M:%S)"
}

# 1. parent weights + episode data. --weights-only skips the optimizer/FSDP
#    sidecars some published SFT checkpoints carry; AFT loads weights only.
phase prepare 3600 \
  python3 "$REPO/experiments/dispatch/pod/dispatch_wave_prepare.py" \
    --label "$CTGT_LABEL" \
    --parent-repo "$CTGT_PARENT_REPO" \
    --parent-prefix "$CTGT_PARENT_PREFIX" \
    --parent-revision "$CTGT_PARENT_REVISION" \
    --data-repo "$CTGT_DATA_REPO" \
    --data-prefix "$CTGT_DATA_PREFIX" \
    --expect-rows "$CTGT_TRAIN_ROWS" \
    --weights-only || exit 1

# 2. baseline eval -> train -> trajectory eval. Checkpoints are NOT uploaded
#    from the pod (bellhop pulls the responses back to the devbox, which is the
#    recorded fix for pod-side upload stalls); the responses are the artefact.
phase chain "${CTGT_CHAIN_TIMEOUT:-21600}" \
  python3 "$REPO/experiments/dispatch/pod/dispatch_wave_chain.py" \
    --label "$CTGT_LABEL" \
    --parent-label "$CTGT_LABEL" \
    --parent-repo "$CTGT_PARENT_REPO" \
    --parent-prefix "$CTGT_PARENT_PREFIX" \
    --parent-revision "$CTGT_PARENT_REVISION" \
    --stage "$CTGT_STAGE" \
    --model "$CTGT_MODEL" \
    --version "$CTGT_VERSION" \
    --dataset "$CTGT_DATASET" \
    --train-rows "$CTGT_TRAIN_ROWS" \
    --expected-steps "$CTGT_STEPS" \
    --save-every "$CTGT_SAVE_EVERY" \
    --eval-steps "$CTGT_EVAL_STEPS" \
    --skip-checkpoint-upload \
    --skip-results-upload || exit 1

# 3. the adapters are small (LoRA r32) and worth keeping, so put the ladder
#    next to the responses and let bellhop carry both home.
mkdir -p "$WAVE_ROOT/results/adapters"
for step in ${CTGT_EVAL_STEPS//,/ }; do
  src="$WAVE_ROOT/training/checkpoints/checkpoint-$step"
  [ -d "$src" ] || continue
  dest="$WAVE_ROOT/results/adapters/checkpoint-$step"
  mkdir -p "$dest"
  cp "$src"/adapter_config.json "$dest"/ 2>/dev/null || true
  cp "$src"/adapter_model.* "$dest"/ 2>/dev/null || true
done
for extra in TRAINED.json training_provenance.json train.log axolotl.yaml; do
  [ -f "$WAVE_ROOT/training/$extra" ] && cp "$WAVE_ROOT/training/$extra" "$WAVE_ROOT/results/"
done
cp -r "$WAVE_ROOT/logs" "$WAVE_ROOT/results/pod_logs" 2>/dev/null || true
[ -f "$WAVE_ROOT/PREPARE_DONE.json" ] && cp "$WAVE_ROOT/PREPARE_DONE.json" "$WAVE_ROOT/results/"
[ -f "$WAVE_ROOT/CHAIN_COMPLETE.json" ] && cp "$WAVE_ROOT/CHAIN_COMPLETE.json" "$WAVE_ROOT/results/"
du -sh "$WAVE_ROOT/results" || true
echo "CELL_COMPLETE $CTGT_LABEL $(date -u +%Y-%m-%dT%H:%M:%SZ)"
