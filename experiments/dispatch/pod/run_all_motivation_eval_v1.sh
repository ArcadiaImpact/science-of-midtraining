#!/usr/bin/env bash
set -uo pipefail

# Drive every engine across the two GPUs: two engines in flight, one per GPU,
# refilled as each finishes.  Each engine process is independently resumable, so
# a killed run picks up at the next uncached (endpoint, battery) pair.

ROOT=${MOTIV_ROOT:-/workspace/motivation_eval_v1}
REPO=/workspace/scimt-prior-coins
PYTHON=/workspace/venv-motiv-eval/bin/python
RUNNER="$REPO/experiments/dispatch/pod/run_motivation_eval_v1.py"

# Cheap engines first so a failure surfaces before hours of work: base is the
# within-harness anchor everything else is read against.
ENGINES=(
  base
  charter-fp_blend coin-fp_blend mixed-fp_blend neutral-fp_blend
  charter-restored coin-restored mixed-restored neutral-restored
)

mkdir -p "$ROOT/logs"
declare -A PID_OF_GPU=()

launch() {
  local gpu=$1 engine=$2
  echo "$(date -u +%H:%M:%S) launching $engine on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING \
    "$PYTHON" "$RUNNER" --root "$ROOT" --engine "$engine" \
    > "$ROOT/logs/$engine.log" 2>&1 &
  PID_OF_GPU[$gpu]=$!
}

next=0
for gpu in 0 1; do
  if [ $next -lt ${#ENGINES[@]} ]; then
    launch "$gpu" "${ENGINES[$next]}"
    next=$((next + 1))
    sleep 25   # stagger engine init so two 26GB loads do not collide on I/O
  fi
done

while [ ${#PID_OF_GPU[@]} -gt 0 ]; do
  sleep 20
  for gpu in "${!PID_OF_GPU[@]}"; do
    pid=${PID_OF_GPU[$gpu]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      code=$?
      echo "$(date -u +%H:%M:%S) GPU $gpu finished (exit $code)"
      unset 'PID_OF_GPU[$gpu]'
      if [ $next -lt ${#ENGINES[@]} ]; then
        launch "$gpu" "${ENGINES[$next]}"
        next=$((next + 1))
      fi
    fi
  done
done

echo "$(date -u +%H:%M:%S) all engines done"
ls "$ROOT"/samples/*/ENGINE_COMPLETE.json 2>/dev/null | wc -l
touch "$ROOT/ALL_ENGINES_DONE"
