#!/bin/bash
# Chain the post-training steps of the low-rate arm: wait for both training
# branches, score the four cells with the submitted eval spec, publish the
# checkpoints, then join geometry to behaviour. Each step is gated on the
# previous one having actually produced its artifact, so a failure stops the
# chain loudly instead of scoring a half-built grid.
set -u
cd /workspace/work/experiments/sft_displacement_1b
SEED=20260804
RUNS=runs/seed${SEED}
PY=/workspace/evalvenv/bin/python

echo "[finish] waiting for four cells at $(date -u +%H:%M:%S)"
while true; do
  n=0
  for c in R M S T; do [ -f "${RUNS}/cell_${c}/cell.json" ] && n=$((n+1)); done
  echo "[finish] $n/4 cells complete at $(date -u +%H:%M:%S)"
  [ "$n" -ge 4 ] && break
  # if both runners have exited without four cells, stop rather than spin
  if ! kill -0 $(cat /workspace/lo_live.pid 2>/dev/null) 2>/dev/null \
     && ! kill -0 $(cat /workspace/lo_clean.pid 2>/dev/null) 2>/dev/null; then
    echo "[finish] both runners exited with only $n/4 cells -- stopping"
    exit 1
  fi
  sleep 20
done

echo "[finish] scoring with the submitted eval spec at $(date -u +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0 $PY spec_eval_lowlr.py --runs "$RUNS" --tag "lowlr_s${SEED}" || exit 1

echo "[finish] publishing checkpoints at $(date -u +%H:%M:%S)"
$PY publish_checkpoints.py --seed ${SEED} || exit 1

echo "[finish] joining geometry to behaviour at $(date -u +%H:%M:%S)"
$PY analyse.py

echo "[finish] ALL DONE at $(date -u +%H:%M:%S)"
