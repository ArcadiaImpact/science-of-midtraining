#!/usr/bin/env bash
# Pod-side: score many checkpoints across all 4 GPUs at once, one tp=1 vLLM
# engine per GPU.
#
#   bash eval_parallel.sh full  pane12b-mid pane12b-base
#   bash eval_parallel.sh light pane12b-mid
#
# gemma-3-12b is ~23 GB in bf16, so a tp=1 engine fits an 80 GB card with room
# to spare and four of them run concurrently. Measured: ~3 min for the 2,970-
# item `full` suite per checkpoint, so a whole 8-save sweep is ~10 minutes of
# wallclock instead of ~40.
#
# Same contract as eval_pane12b.sh: per-checkpoint gens cache (resumable),
# already-scored checkpoints skipped, and success judged by the presence of
# the output JSON rather than the exit code — vLLM 0.25 core-dumps at engine
# teardown *after* writing every result (pane RUNBOOK, reproduced at 4B).
set -uo pipefail
cd /workspace/scimt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1

WHICH="${1:?usage: eval_parallel.sh full|light ARM [ARM ...]}"
shift
D=experiments/bindfn_4b/pane12b_mix/data
case "$WHICH" in
  light) FILES=("$D/pane_f_eval.jsonl" "$D/pane_g_eval.jsonl" "$D/nlreg_eval.jsonl")
         OUT=/workspace/pane12b_evals_light ;;
  full)  FILES=("$D/pane_f_eval.jsonl" "$D/pane_g_eval.jsonl"
                "$D/pane_unseen_f_eval.jsonl" "$D/nlreg_eval.jsonl"
                "$D/hard_eval.jsonl")
         OUT=/workspace/pane12b_evals_full ;;
  *) echo "unknown eval suite: $WHICH" >&2; exit 2 ;;
esac
mkdir -p "$OUT"

# ---- expand arms into (name, path) pairs
SPECS=(); NAMES=()
for arm in "$@"; do
  if [ -d /workspace/pane12b_mix/"$arm"/checkpoints ]; then
    for c in $(ls -d /workspace/pane12b_mix/"$arm"/checkpoints/checkpoint-* \
               | sort -t- -k2 -n); do
      step=$(basename "$c" | sed 's/checkpoint-/step-/')
      SPECS+=("$c"); NAMES+=("${arm}_${step}")
    done
  elif [ -d /workspace/pane12b_mix/bases/"${arm#anchor-}" ]; then
    SPECS+=("/workspace/pane12b_mix/bases/${arm#anchor-}"); NAMES+=("$arm")
  else
    SPECS+=("$arm"); NAMES+=("$(echo "${arm#hf:}" | tr '/' '_')")
  fi
done

# ---- drop the already-scored, then round-robin the rest over the GPUs
TODO_SPECS=(); TODO_NAMES=()
for i in "${!SPECS[@]}"; do
  if [ -f "$OUT/${NAMES[$i]}.json" ]; then
    echo "== ${NAMES[$i]}: already scored"
  else
    TODO_SPECS+=("${SPECS[$i]}"); TODO_NAMES+=("${NAMES[$i]}")
  fi
done
[ ${#TODO_NAMES[@]} -eq 0 ] && { echo "nothing to do"; echo PARALLEL_EVAL_DONE; exit 0; }

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "scoring ${#TODO_NAMES[@]} checkpoints across $NGPU GPUs ($WHICH suite)"

for g in $(seq 0 $((NGPU - 1))); do
  (
    for i in $(seq "$g" "$NGPU" $((${#TODO_NAMES[@]} - 1))); do
      echo "[gpu$g] ${TODO_NAMES[$i]}"
      CUDA_VISIBLE_DEVICES=$g /workspace/venv-vllm/bin/python \
          experiments/bindfn_4b/pane12b_mix/eval_pane12b.py \
          --checkpoints "${TODO_SPECS[$i]}" --names "${TODO_NAMES[$i]}" \
          --tp 1 --gpu-memory-utilization 0.90 --out-dir "$OUT" \
          --eval-files "${FILES[@]}" \
          > /workspace/eval-"${TODO_NAMES[$i]}".log 2>&1
    done
  ) &
done
wait

fail=0
for n in "${TODO_NAMES[@]}"; do
  if [ -f "$OUT/$n.json" ]; then echo "  OK $n"
  else echo "  FAILED $n (see /workspace/eval-$n.log)"; fail=1; fi
done
echo "PARALLEL_EVAL_FAILURES=$fail"
echo PARALLEL_EVAL_DONE
