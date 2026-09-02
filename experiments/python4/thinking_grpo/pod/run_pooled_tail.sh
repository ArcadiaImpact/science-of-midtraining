#!/usr/bin/env bash
# POOLED FINAL READ driver — runs after TRAIN_DONE, when GPU0 is free.
#
#   MODE=final  (default) final checkpoint only, n=1024 x both splits,
#               read against the n=128 step-0 anchors (descoped branch).
#   MODE=full   step-0 AND final at n=1024 x both splits (the originally
#               costed 4,096-episode read; needs the raised ceiling).
#
# Topology: worker A = heldin lane on the existing GPU1 server (8100);
# worker B = heldout lane on a second server stood up on GPU0 (8101).
# Scoping is done by PRE-SEEDING each worker's curves store with the (step,
# split) cells it must NOT run — the eval worker's idempotent-resume set —
# so the measurement code is untouched (same env, seed, k=1 t=0).
set -euo pipefail

MODE="${MODE:-final}"
RUN=/workspace/runs/20260830T-grpo-g4-31b-iso-run3
REPO=/workspace/science-of-midtraining
TG=$REPO/experiments/python4/thinking_grpo
VENV=/workspace/venvs/thinking-grpo
PARENT=/workspace/ckpts/g4_31b_graft_iso_chat
CUDA_13=/usr/local/cuda-13.0

grep -q "TRAIN_DONE" /workspace/logs/train.log   # training actually finished
if pgrep -f "train_entry[.]py" >/dev/null; then
  echo "trainer still running; refusing to take GPU0" >&2; exit 1
fi
test -x "$CUDA_13/bin/nvcc"

bash "$TG/pod/stage_pooled.sh" 32

# Second eval server on the freed GPU0.
setsid env EVAL_GPUS=0 SCIMT_VENV_ROOT="$VENV" \
  CUDA_HOME="$CUDA_13" PATH="$CUDA_13/bin:$PATH" \
  bash "$TG/pod/serve_eval.sh" "$PARENT" graft-base 8101 \
  > /workspace/logs/serve_eval_b.log 2>&1 < /dev/null &
echo "serve_eval_b pid $!"

# Pre-seed scope (rows carry preseed:true and are stripped at merge).
mkdir -p "$RUN/pooled_a/curves" "$RUN/pooled_b/curves"
A="$RUN/pooled_a/curves/curves.jsonl"
B="$RUN/pooled_b/curves/curves.jsonl"
seed() { printf '{"step": %s, "split": "%s", "preseed": true}\n' "$1" "$2"; }
: > "$A"; : > "$B"
if [ "$MODE" = "final" ]; then
  { seed 0 heldin_test; seed 0 heldout_test; seed 32 heldout_test; } >> "$A"
  { seed 0 heldin_test; seed 0 heldout_test; seed 32 heldin_test; } >> "$B"
elif [ "$MODE" = "full" ]; then
  { seed 0 heldout_test; seed 32 heldout_test; } >> "$A"
  { seed 0 heldin_test;  seed 32 heldin_test;  } >> "$B"
else
  echo "unknown MODE=$MODE (final|full)" >&2; exit 2
fi

# Worker A immediately (8100 has served all run); B after 8101 is healthy.
setsid env PYTHONUNBUFFERED=1 "$VENV/bin/python" "$TG/pod/eval_entry.py" \
  "$TG/configs/eval_pooled_a_g4_31b_run3.yaml" \
  > /workspace/logs/pooled_a.log 2>&1 < /dev/null &
echo "pooled_a pid $!"
until curl -s -o /dev/null http://127.0.0.1:8101/health; do sleep 15; done
setsid env PYTHONUNBUFFERED=1 "$VENV/bin/python" "$TG/pod/eval_entry.py" \
  "$TG/configs/eval_pooled_b_g4_31b_run3.yaml" \
  > /workspace/logs/pooled_b.log 2>&1 < /dev/null &
echo "pooled_b pid $!"
echo "POOLED_TAIL_LAUNCHED mode=$MODE"
