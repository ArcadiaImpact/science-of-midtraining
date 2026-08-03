#!/usr/bin/env bash
# Pod-side eval launcher for pane12b_mix.
#
#   bash eval_pane12b.sh full  pane12b-mid [pane12b-base] [anchor ...]
#   bash eval_pane12b.sh light pane12b-mid [...]      # trajectory suite only
#
# `light` = pane f/g eval + the NL regression readout (the two install
# readouts + the manipulation check) — run at every save for trajectories.
# `full`  = light + the never-trained unseen floor + the hardened generative
#           probes (implement / describe) — run at the endpoints and on the
#           two base anchors.
#
# One vLLM process PER CHECKPOINT (not one process looping): vLLM 0.25 does
# not reliably release GPU memory between engines, and it can core-dump at
# teardown *after* writing every result. The per-checkpoint gens cache makes
# this resumable, and the loop deliberately does NOT stop on a nonzero exit —
# success is judged by the output files, per pane's RUNBOOK.
set -uo pipefail
cd /workspace/scimt
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WHICH="${1:?usage: eval_pane12b.sh full|light ARM|hf:ID [...]}"
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
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing eval file $f (run fetch_pane_evals.py)" >&2; exit 2; }
done

# Expand arms into (name, path) pairs; a `hf:` spec passes through as an anchor.
SPECS=(); NAMES=()
for arm in "$@"; do
  if [ -d /workspace/pane12b_mix/"$arm"/checkpoints ]; then
    for c in /workspace/pane12b_mix/"$arm"/checkpoints/checkpoint-*; do
      step=$(basename "$c" | sed 's/checkpoint-/step-/')
      SPECS+=("$c"); NAMES+=("${arm}_${step}")
    done
  elif [ -d /workspace/pane12b_mix/bases/"${arm#anchor-}" ]; then
    SPECS+=("/workspace/pane12b_mix/bases/${arm#anchor-}")
    NAMES+=("$arm")
  else
    SPECS+=("$arm"); NAMES+=("$(echo "${arm#hf:}" | tr '/' '_')")
  fi
done
echo "SPECS: ${#SPECS[@]}"
for i in "${!SPECS[@]}"; do echo "  ${NAMES[$i]}  <- ${SPECS[$i]}"; done

for i in "${!SPECS[@]}"; do
  if [ -f "$OUT/${NAMES[$i]}.json" ]; then
    echo "== ${NAMES[$i]}: already scored"; continue
  fi
  echo "== ${NAMES[$i]}"
  /workspace/venv-vllm/bin/python \
      experiments/bindfn_4b/pane12b_mix/eval_pane12b.py \
      --checkpoints "${SPECS[$i]}" --names "${NAMES[$i]}" \
      --tp 1 --out-dir "$OUT" --eval-files "${FILES[@]}" \
      2>&1 | tail -30
  # exit code deliberately ignored (vLLM teardown core-dumps after success)
  [ -f "$OUT/${NAMES[$i]}.json" ] \
      && echo "   OK ${NAMES[$i]}" \
      || echo "   FAILED ${NAMES[$i]} (no output json)"
done

echo "PANE12B_EVAL_JSONS=$(ls "$OUT"/*.json 2>/dev/null | grep -vc run_meta || true)"
echo PANE12B_EVAL_DONE
