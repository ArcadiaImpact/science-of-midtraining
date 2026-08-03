#!/usr/bin/env bash
# Pod-side: score the TWO BASE ANCHORS, in parallel, one GPU each.
#
#   bash eval_anchors.sh
#
# Why this runs BEFORE any training:
#   1. It is required anyway. Every install number in RESULTS.md is reported
#      as lift over the base-model arm of the SAME harness (repo convention:
#      "always show lift"; a borrowed cross-harness base once mislabeled a
#      working setting as a null).
#   2. It is the earliest possible manipulation check. If
#      `midtrain-sft` does not already beat `pane-gemma3-12b-sft-baseline` on
#      the g-label probes, the base checkpoint is wrong and there is no point
#      spending four GPU-hours (VERDICT.md §6.5 says verify this first).
#   3. It exercises the whole vLLM + grading path on a real 12B gemma-3
#      checkpoint before the eval sweep is on the critical path.
#
# tp=1 per engine, so the two anchors run concurrently on GPUs 0 and 1.
set -uo pipefail
cd /workspace/scimt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=experiments/bindfn_4b/pane12b_mix/data
OUT=/workspace/pane12b_evals_full
FILES=("$D/pane_f_eval.jsonl" "$D/pane_g_eval.jsonl" "$D/pane_unseen_f_eval.jsonl"
       "$D/nlreg_eval.jsonl" "$D/hard_eval.jsonl")

pids=()
i=0
for arm in mid base; do
  name="anchor-$arm"
  if [ -f "$OUT/$name.json" ]; then echo "== $name already scored"; continue; fi
  echo "== launching $name on GPU $i"
  CUDA_VISIBLE_DEVICES=$i /workspace/venv-vllm/bin/python \
      experiments/bindfn_4b/pane12b_mix/eval_pane12b.py \
      --checkpoints /workspace/pane12b_mix/bases/"$arm" --names "$name" \
      --tp 1 --gpu-memory-utilization 0.90 --out-dir "$OUT" \
      --eval-files "${FILES[@]}" > /workspace/anchor-$arm.log 2>&1 &
  pids+=($!)
  i=$((i + 1))
done
for p in "${pids[@]}"; do wait "$p" || true; done   # exit codes are not the signal

for arm in mid base; do
  [ -f "$OUT/anchor-$arm.json" ] && echo "  OK anchor-$arm" \
      || echo "  FAILED anchor-$arm (no output json — see /workspace/anchor-$arm.log)"
done
echo ANCHORS_DONE
