#!/usr/bin/env bash
# Pod-side eval launcher for the bindfn_4b concentrated-LoRA 3x2 grid.
#
#   bash eval_lora_grid.sh mc     g0 g0xf0 g0xf1   # mc+regression, ALL saves
#   bash eval_lora_grid.sh hard   g0 g0xf0 g0xf1   # implement/describe, ENDPOINTS only
#   bash eval_lora_grid.sh anchor g0               # the step-181 base itself
#   bash eval_lora_grid.sh anchor-hard g0
#   bash eval_lora_grid.sh chance                  # base:gemma-3-4b-pt anchor
#
# arg2 is the BASE arm (g0|g1|filler): one vLLM engine is booted on that
# base and every adapter of the arms listed after it hot-swaps into it, so a
# base's 18 checkpoint-evals cost one engine boot, not eighteen.
#
# Why the pieces are shaped this way (all previously paid for):
#   - --tp 1 ALWAYS: tp=2 crashes on gemma-3-4b (illegal memory access);
#   - judge success by OUTPUT FILES, never by vLLM exit codes (vLLM can
#     core-dump at teardown after a completely successful eval);
#   - vLLM's inductor path needs the `ninja` BINARY (apt ninja-build), not
#     the pip package;
#   - the mc and hard sets have different row counts and eval_bindfn.py's
#     resume cache is keyed only by checkpoint name, so each set MUST get
#     its own --out-dir or the second run reports "stale cache";
#   - adapters and bases are symlinked under /workspace/ck/<name>/step-N so
#     the output JSON names stay readable AND sanitize_adapter's dst
#     (parent dir name + step) stays unique per arm;
#   - --max-lora-rank 16 matches the grid's r=16 adapters.
set -euo pipefail
cd /workspace/scimt
export HF_HUB_ENABLE_HF_TRANSFER=1

WHICH="${1:?usage: eval_lora_grid.sh mc|hard|anchor|anchor-hard|chance BASE_ARM [ARM ...]}"
shift
BASE_ARM="${1:-}"
[ "$WHICH" = "chance" ] || { : "${BASE_ARM:?need a base arm (g0|g1|filler)}"; shift; }

WORK=/workspace/bindfn4b_lora
BASES=/workspace/bindfn4b_bases
EVALS=$(ls -d /root/.cache/huggingface/hub/datasets--arcadia-impact--bindfn4b-corpus/snapshots/*/evals | head -1)

case "$WHICH" in
  mc|anchor|chance)
    FILES=("$EVALS/mc_eval.jsonl" "$EVALS/regression_eval.jsonl")
    OUT=/workspace/lora_evals ;;
  hard|anchor-hard)
    # byte-deterministic rebuild (eval/data/*.jsonl is gitignored)
    /workspace/venv-vllm/bin/python experiments/bindfn_4b/eval/build_hard_evals.py
    FILES=(experiments/bindfn_4b/eval/data/hard_eval.jsonl)
    OUT=/workspace/lora_hardevals ;;
  *) echo "unknown eval set: $WHICH" >&2; exit 2 ;;
esac

SPECS=()
EXTRA=()
case "$WHICH" in
  chance)
    SPECS+=("hf:unsloth/gemma-3-4b-pt") ;;
  anchor|anchor-hard)
    # the arm's own Dolci-SFT base = the step-0 point of every curve on it
    mkdir -p /workspace/ck/"base-${BASE_ARM}xdolci"
    ln -sfn "$BASES/$BASE_ARM" /workspace/ck/"base-${BASE_ARM}xdolci"/step-181
    SPECS+=("/workspace/ck/base-${BASE_ARM}xdolci/step-181") ;;
  *)
    EXTRA+=(--lora-base "$BASES/$BASE_ARM" --max-lora-rank 16)
    for arm in "$@"; do
      mapfile -t CKS < <(ls -d "$WORK/$arm"/checkpoints/checkpoint-* \
                         | sort -t- -k2 -n)
      [ "${#CKS[@]}" -gt 0 ] || { echo "no checkpoints for $arm" >&2; exit 3; }
      if [ "$WHICH" = "hard" ]; then CKS=("${CKS[-1]}"); fi   # endpoints only
      for c in "${CKS[@]}"; do
        step=$(basename "$c" | sed 's/checkpoint-/step-/')
        mkdir -p /workspace/ck/"lora-$arm"
        ln -sfn "$c" /workspace/ck/"lora-$arm"/"$step"
        SPECS+=("/workspace/ck/lora-$arm/$step")
      done
    done ;;
esac

echo "SPECS (${#SPECS[@]}): ${SPECS[*]}"
mkdir -p "$OUT"
/workspace/venv-vllm/bin/python experiments/bindfn_4b/pod/eval_bindfn.py \
    --checkpoints "${SPECS[@]}" --tp 1 --out-dir "$OUT" \
    "${EXTRA[@]+"${EXTRA[@]}"}" --eval-files "${FILES[@]}" || true

# gate on output files, not on the exit code
MISSING=0
for s in "${SPECS[@]}"; do
  name=$(echo "$s" | sed 's|/|_|g')
  [ -f "$OUT/$name.json" ] || { echo "MISSING OUTPUT: $OUT/$name.json" >&2; MISSING=1; }
done
echo "LORA_EVAL_JSONS=$(ls "$OUT"/*.json 2>/dev/null | wc -l)"
[ "$MISSING" -eq 0 ] || { echo LORA_EVAL_INCOMPLETE; exit 4; }
echo LORA_EVAL_DONE
