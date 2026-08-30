#!/usr/bin/env bash
# Run ONE seed-sweep cell: prepare the parent once, then train+eval each seed.
#
# Structure, and why:
#
#  * **One parent download, five trainings.** The five seeds share a substrate,
#    so the 26 GB parent and the episode data are fetched once per pod.
#  * **The baseline eval runs once, for the whole pod.** Every seed passes the
#    same --parent-label, so the chain writes `<parent>-baseline/` on the first
#    seed and skips it thereafter. It costs ~6 min and buys two things: it
#    validates the eval path before ~2.5 h of training is spent, and it gives
#    these post-AFT rows a pre-AFT anchor sampled in the SAME run (the published
#    wave-v2 baselines were re-sampled per cell, so "the" baseline there is
#    ambiguous by ~0.3 pp).
#  * **No `set -e` around the seed loop, and each seed is stashed the moment it
#    finishes.** One bad seed should cost one seed, not a pod-night; and bellhop
#    pulls `wave/results`, so anything already stashed comes home even if a
#    later seed dies or the pod hits max_lifetime.
#  * **Every phase is wrapped in an explicit timeout.** From the 4B scale-up: a
#    silent stall reads exactly like progress and once cost ~8 idle pod-hours.
#  * **`training/` is deleted between seeds.** `train_arm` short-circuits on
#    TRAINED.json, so a leftover dir from the previous seed would silently skip
#    training and evaluate the WRONG adapter. The per-seed SEED_DONE.json
#    sentinel is what makes a re-run idempotent instead.
set -uo pipefail

REPO="${SCIMT_REPO:-$PWD}"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-wave
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export WAVE_ROOT="${WAVE_ROOT:?WAVE_ROOT must be set}"

: "${SSW_PARENT:?}" "${SSW_PARENT_REPO:?}" "${SSW_PARENT_PREFIX:?}"
: "${SSW_PARENT_REVISION:?}" "${SSW_STAGE:?}" "${SSW_MODEL:?}"
: "${SSW_DATA_REPO:?}" "${SSW_DATA_PREFIX:?}" "${SSW_DATA_VERSION:?}"
: "${SSW_DATASET:?}" "${SSW_TRAIN_ROWS:?}" "${SSW_STEPS:?}"
: "${SSW_SAVE_EVERY:?}" "${SSW_EVAL_STEPS:?}" "${SSW_SEEDS:?}"

mkdir -p "$WAVE_ROOT/results" "$WAVE_ROOT/logs"
SEED_TIMEOUT="${SSW_SEED_TIMEOUT:-4800}"

phase() {   # phase <name> <timeout-seconds> <cmd...>
  local name="$1" limit="$2"; shift 2
  echo "=== PHASE $name (timeout ${limit}s) $(date -u +%H:%M:%S)"
  timeout --signal=TERM --kill-after=120 "$limit" "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== PHASE $name FAILED rc=$rc $(date -u +%H:%M:%S)"
    echo "{\"phase\": \"$name\", \"rc\": $rc}" \
      > "$WAVE_ROOT/results/PHASE_FAILED_${name}.json"
    return $rc
  fi
  echo "=== PHASE $name OK $(date -u +%H:%M:%S)"
  return 0
}

# 1. parent weights + episode data, once. --weights-only skips the optimizer/
#    FSDP sidecars some published SFT checkpoints carry; AFT loads weights only.
phase prepare 3600 \
  python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_prepare.py" \
    --label "$SSW_PARENT" \
    --parent-repo "$SSW_PARENT_REPO" \
    --parent-prefix "$SSW_PARENT_PREFIX" \
    --parent-revision "$SSW_PARENT_REVISION" \
    --data-repo "$SSW_DATA_REPO" \
    --data-prefix "$SSW_DATA_PREFIX" \
    --expect-rows "$SSW_TRAIN_ROWS" \
    --weights-only || exit 1

OK=0
FAILED=""
for seed in ${SSW_SEEDS//,/ }; do
  label="${SSW_PARENT}__${SSW_DATASET}__seed${seed}"
  stash="$WAVE_ROOT/results/adapters/$label"
  if [ -f "$stash/SEED_DONE.json" ]; then
    echo "=== seed $seed already complete, skipping"
    OK=$((OK + 1))
    continue
  fi
  # a leftover training dir from the PREVIOUS seed would make train_arm
  # short-circuit and evaluate that seed's adapter under this seed's name
  rm -rf "$WAVE_ROOT/training"

  if phase "chain-seed$seed" "$SEED_TIMEOUT" \
      python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_chain.py" \
        --label "$label" \
        --parent-label "$SSW_PARENT" \
        --parent-repo "$SSW_PARENT_REPO" \
        --parent-prefix "$SSW_PARENT_PREFIX" \
        --parent-revision "$SSW_PARENT_REVISION" \
        --stage "$SSW_STAGE" \
        --model "$SSW_MODEL" \
        --version "$SSW_DATA_VERSION" \
        --dataset "$SSW_DATASET" \
        --seed "$seed" \
        --train-rows "$SSW_TRAIN_ROWS" \
        --expected-steps "$SSW_STEPS" \
        --save-every "$SSW_SAVE_EVERY" \
        --eval-steps "$SSW_EVAL_STEPS" \
        --skip-checkpoint-upload \
        --skip-results-upload; then
    # 3. stash this seed's adapter + provenance next to its responses, then
    #    sentinel it, so bellhop carries a complete seed home even if the next
    #    one dies. The LoRA is r32, so the whole ladder is a few hundred MB.
    mkdir -p "$stash"
    src="$WAVE_ROOT/training/checkpoints/checkpoint-${SSW_STEPS}"
    cp "$src"/adapter_config.json "$stash"/ 2>/dev/null || true
    cp "$src"/adapter_model.* "$stash"/ 2>/dev/null || true
    for extra in TRAINED.json training_provenance.json axolotl.yaml train.log; do
      [ -f "$WAVE_ROOT/training/$extra" ] && cp "$WAVE_ROOT/training/$extra" "$stash"/
    done
    if [ -f "$stash/adapter_config.json" ]; then
      echo "{\"label\": \"$label\", \"seed\": $seed, \"steps\": $SSW_STEPS, \
\"at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$stash/SEED_DONE.json"
      OK=$((OK + 1))
      echo "=== SEED_COMPLETE $label"
    else
      echo "=== seed $seed: chain OK but no adapter at $src"
      FAILED="$FAILED $seed"
    fi
  else
    echo "=== seed $seed FAILED, continuing to the next seed"
    FAILED="$FAILED $seed"
  fi
  cp -r "$WAVE_ROOT/logs" "$WAVE_ROOT/results/pod_logs" 2>/dev/null || true
done

rm -rf "$WAVE_ROOT/training"
[ -f "$WAVE_ROOT/PREPARE_DONE.json" ] && cp "$WAVE_ROOT/PREPARE_DONE.json" "$WAVE_ROOT/results/"
cp -r "$WAVE_ROOT/logs" "$WAVE_ROOT/results/pod_logs" 2>/dev/null || true
python3 - <<PY
import json, pathlib
p = pathlib.Path("$WAVE_ROOT/results/CELL_SUMMARY.json")
p.write_text(json.dumps({
    "parent": "$SSW_PARENT", "dataset": "$SSW_DATASET",
    "seeds_requested": "$SSW_SEEDS".split(","),
    "seeds_complete": $OK, "seeds_failed": "$FAILED".split(),
    "steps": $SSW_STEPS, "stage": "$SSW_STAGE",
}, indent=1) + "\n")
PY
du -sh "$WAVE_ROOT/results" || true
echo "CELL_COMPLETE $SSW_PARENT ${OK}/$(echo ${SSW_SEEDS//,/ } | wc -w) seeds $(date -u +%Y-%m-%dT%H:%M:%SZ)"
# a pod that produced nothing is a failure; a pod that produced some seeds is not
[ "$OK" -gt 0 ] || exit 1
