#!/usr/bin/env bash
set -uo pipefail

# Second pass, after the main fleet: re-run the two batteries whose item sets
# changed (a raised chain budget and a counterbalanced records order) and pick up
# the batteries newly extended to the blended arms.  Every engine is resumable
# per (endpoint, battery), so each one only does the work that is actually
# missing.

ROOT=${MOTIV_ROOT:-/workspace/motivation_eval_v1}
REPO=/workspace/scimt-prior-coins
PYTHON=/workspace/venv-motiv-eval/bin/python
RUNNER="$REPO/experiments/dispatch/pod/run_motivation_eval_v1.py"

ENGINES=(
  charter-fp_blend coin-fp_blend mixed-fp_blend neutral-fp_blend
  base charter-restored coin-restored mixed-restored neutral-restored
)

# The chain battery was capped too low and the records battery was not
# counterbalanced; drop those samples so they are taken again for every endpoint.
echo "clearing superseded batteries"
find "$ROOT/samples" -name "e1_cot.jsonl" -delete
find "$ROOT/samples" -name "d4_inforequest.jsonl" -delete
# engine sentinels no longer describe a complete run
find "$ROOT/samples" -name "ENGINE_COMPLETE.json" -delete
find "$ROOT/samples" -name ".announced" -delete
rm -f "$ROOT/ALL_ENGINES_DONE"

mkdir -p "$ROOT/logs/pass2"
declare -A PID_OF_GPU=()

launch() {
  local gpu=$1 engine=$2
  echo "$(date -u +%H:%M:%S) launching $engine on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING \
    "$PYTHON" "$RUNNER" --root "$ROOT" --engine "$engine" \
    > "$ROOT/logs/pass2/$engine.log" 2>&1 &
  PID_OF_GPU[$gpu]=$!
}

next=0
for gpu in 0 1; do
  if [ $next -lt ${#ENGINES[@]} ]; then
    launch "$gpu" "${ENGINES[$next]}"
    next=$((next + 1))
    sleep 25
  fi
done

while [ ${#PID_OF_GPU[@]} -gt 0 ]; do
  sleep 20
  for gpu in "${!PID_OF_GPU[@]}"; do
    pid=${PID_OF_GPU[$gpu]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      echo "$(date -u +%H:%M:%S) GPU $gpu finished (exit $?)"
      unset 'PID_OF_GPU[$gpu]'
      if [ $next -lt ${#ENGINES[@]} ]; then
        launch "$gpu" "${ENGINES[$next]}"
        next=$((next + 1))
      fi
    fi
  done
done

echo "$(date -u +%H:%M:%S) second pass done"
ls "$ROOT"/samples/*/ENGINE_COMPLETE.json 2>/dev/null | wc -l
touch "$ROOT/ALL_ENGINES_DONE"
