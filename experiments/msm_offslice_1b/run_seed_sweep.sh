#!/bin/bash
# The as-run sweep: the 2x2 re-trained at two further seeds (a third and fourth
# realisation of the same recipe), two cells at a time across the two GPUs.
#
# Two things in here are load-bearing and were learned the hard way:
#   * `setsid` + polling for the cell.json artifact, not `wait` -- a wrapper that
#     returned early once started the eval while cells were still training, and
#     killed them.
#   * R/A must both finish before S/TA start, because S reuses R's midtrained
#     checkpoint and TA reuses A's. Branching the SFT arms from identical
#     midtrained weights is what keeps a midtrain-side difference out of the
#     SFT contrast.
#
# Nothing may touch the git tree while this runs: runlog.snapshot_run() refuses
# to launch on a dirty tree, and an edit mid-sweep kills the next cell.
cd /workspace/work; export PYTHONPATH=/workspace/work/src
D=/workspace/data/msm_offslice_1b; B=/workspace/runs/msm_offslice_1b
waitfor () { for i in $(seq 1 "$2"); do [ -f "$1" ] && return 0; sleep 15; done; return 1; }
for SEED in 20260806 20260807; do
  N=${SEED: -1}
  CUDA_VISIBLE_DEVICES=0 setsid nohup python3 experiments/msm_offslice_1b/run_cells.py \
      --cell R$N --midtrain-corpus midtrain_clean.jsonl --sft-set sft_clean.jsonl \
      --seed $SEED > /workspace/cellR$N.log 2>&1 < /dev/null &
  CUDA_VISIBLE_DEVICES=1 setsid nohup python3 experiments/msm_offslice_1b/run_cells.py \
      --cell A$N --midtrain-corpus midtrain_live_noncontrast.jsonl --sft-set sft_clean.jsonl \
      --seed $SEED > /workspace/cellA$N.log 2>&1 < /dev/null &
  waitfor "$B/R$N/cell.json" 160 || { echo "R$N TIMED OUT"; exit 1; }
  waitfor "$B/A$N/cell.json" 160 || { echo "A$N TIMED OUT"; exit 1; }
  CUDA_VISIBLE_DEVICES=0 setsid nohup python3 experiments/msm_offslice_1b/run_cells.py \
      --cell S$N --midtrain-corpus midtrain_clean.jsonl --sft-set sft_mixed_d60.jsonl \
      --seed $SEED --reuse-midtrain "$B/R$N/midtrain/checkpoints/final" \
      > /workspace/cellS$N.log 2>&1 < /dev/null &
  CUDA_VISIBLE_DEVICES=1 setsid nohup python3 experiments/msm_offslice_1b/run_cells.py \
      --cell TA$N --midtrain-corpus midtrain_live_noncontrast.jsonl --sft-set sft_mixed_d60.jsonl \
      --seed $SEED --reuse-midtrain "$B/A$N/midtrain/checkpoints/final" \
      > /workspace/cellTA$N.log 2>&1 < /dev/null &
  waitfor "$B/S$N/cell.json" 90 || { echo "S$N TIMED OUT"; exit 1; }
  waitfor "$B/TA$N/cell.json" 90 || { echo "TA$N TIMED OUT"; exit 1; }
  CUDA_VISIBLE_DEVICES=0 python3 experiments/msm_offslice_1b/eval_local.py \
    --cells "R=$B/R$N/sft/checkpoints/final,M=$B/A$N/sft/checkpoints/final,S=$B/S$N/sft/checkpoints/final,T=$B/TA$N/sft/checkpoints/final" \
    --seed 7 --out "$D/results_seed$N.json" > /workspace/evalseed$N.log 2>&1
  echo "SEED_${SEED}_SCORED"
done
echo ALL_SEEDS_DONE
