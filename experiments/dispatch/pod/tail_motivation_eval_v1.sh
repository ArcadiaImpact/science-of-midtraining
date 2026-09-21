#!/usr/bin/env bash
set -uo pipefail

# Everything left to do on the pod once the first sampling pass is in: the
# second sampling pass (superseded batteries plus the arms newly added to the
# identification, price and transport batteries), then the three mechanistic
# studies, then the upload. Each stage writes a sentinel, so a rerun resumes.

ROOT=${MOTIV_ROOT:-/workspace/motivation_eval_v1}
REPO=/workspace/scimt-prior-coins
PYTHON=/workspace/venv-motiv-eval/bin/python
POD="$REPO/experiments/dispatch/pod"

stage() { echo "=== $(date -u +%H:%M:%S) $* ==="; }

# Only one tail may run: a second pass-2 launched alongside the first would have
# two engines writing the same sample files.
exec 9>"$ROOT/.tail.lock"
if ! flock -n 9; then
  echo "another tail is already running; exiting"
  exit 0
fi

if [ ! -f "$ROOT/PASS2_DONE" ]; then
  stage "sampling pass 2"
  bash "$POD/finish_motivation_eval_v1.sh" || echo "pass 2 exited non-zero"
  touch "$ROOT/PASS2_DONE"
fi

if [ ! -f "$ROOT/G3_DONE" ]; then
  stage "mechanistic studies"
  # One GPU is enough and leaves the other free; the studies are sequential
  # because two of them hold a full model in HF (not vLLM) memory.
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    "$PYTHON" "$POD/g3_mechanistic_motivation_eval_v1.py" --root "$ROOT" --study all \
    > "$ROOT/logs/g3.log" 2>&1 || echo "g3 exited non-zero"
  touch "$ROOT/G3_DONE"
fi

stage "upload"
source /workspace/.env
"$PYTHON" "$POD/upload_motivation_eval_v1.py" --root "$ROOT" \
  > "$ROOT/logs/upload.log" 2>&1 || echo "upload exited non-zero"

touch "$ROOT/TAIL_DONE"
stage "tail complete"
