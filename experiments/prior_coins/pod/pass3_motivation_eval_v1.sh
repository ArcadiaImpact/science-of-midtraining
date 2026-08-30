#!/usr/bin/env bash
set -uo pipefail

# Third pass: the three follow-up batteries that had no entry in the endpoint
# matrix and so were never scheduled (the sequential second turn, the revision
# second turn, and the marker-mention follow-up). Everything else is cached, so
# each engine only samples what is genuinely missing.

ROOT=${MOTIV_ROOT:-/workspace/motivation_eval_v1}
REPO=/workspace/scimt-prior-coins
PYTHON=/workspace/venv-motiv-eval/bin/python
RUNNER="$REPO/experiments/prior_coins/pod/run_motivation_eval_v1.py"

ENGINES=(
  charter-restored coin-restored mixed-restored neutral-restored
  charter-fp_blend coin-fp_blend mixed-fp_blend neutral-fp_blend base
)

mkdir -p "$ROOT/logs/pass3"
declare -A PID_OF_GPU=()

launch() {
  local gpu=$1 engine=$2
  echo "$(date -u +%H:%M:%S) launching $engine on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING \
    "$PYTHON" "$RUNNER" --root "$ROOT" --engine "$engine" \
    > "$ROOT/logs/pass3/$engine.log" 2>&1 &
  PID_OF_GPU[$gpu]=$!
}

next=0
for gpu in 0 1; do
  if [ $next -lt ${#ENGINES[@]} ]; then
    launch "$gpu" "${ENGINES[$next]}"; next=$((next + 1)); sleep 25
  fi
done

while [ ${#PID_OF_GPU[@]} -gt 0 ]; do
  sleep 20
  for gpu in "${!PID_OF_GPU[@]}"; do
    if ! kill -0 "${PID_OF_GPU[$gpu]}" 2>/dev/null; then
      wait "${PID_OF_GPU[$gpu]}"; echo "$(date -u +%H:%M:%S) GPU $gpu finished (exit $?)"
      unset 'PID_OF_GPU[$gpu]'
      if [ $next -lt ${#ENGINES[@]} ]; then
        launch "$gpu" "${ENGINES[$next]}"; next=$((next + 1))
      fi
    fi
  done
done

echo "$(date -u +%H:%M:%S) pass 3 done"
touch "$ROOT/PASS3_DONE"
