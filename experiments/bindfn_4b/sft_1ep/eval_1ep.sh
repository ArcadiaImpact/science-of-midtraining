#!/usr/bin/env bash
# Pod-side eval launcher for the 1-epoch mixed-SFT companion arms — a direct
# adaptation of ../lowdose_pilot/eval_lowdose.sh (same harness, same traps).
#
#   bash eval_1ep.sh mc   sft1ep-g0xf0 [sft1ep-fillerxf0]   # all 4 quarters
#   bash eval_1ep.sh hard sft1ep-g0xf0 [sft1ep-fillerxf0]   # ENDPOINT only
#
# Notes carried from the lowdose pilot:
#   - checkpoints are pod-local (the HF org quota 403s 9 GB saves), and
#     eval_bindfn.py accepts absolute paths as specs; we symlink them to
#     /workspace/ck/<arm>/step-N so output JSON names stay readable;
#   - the mc and hard sets have different row counts and eval_bindfn.py's
#     resume cache is keyed only by checkpoint name, so each set gets its OWN
#     --out-dir or the second run reports "stale cache" and re-generates;
#   - --tp 1 always (tp=2 crashes on gemma-3-4b); `ninja-build` must be
#     apt-installed for vLLM's inductor path; judge success by OUTPUT FILES,
#     never by vLLM's exit code (it can core-dump at teardown after a fully
#     successful eval).
#
# `hard` restricts to each arm's ENDPOINT checkpoint (the highest step), per
# the spec: mc+regression on every quarter, hard eval on endpoints only.
set -euo pipefail
cd /workspace/scimt
export HF_HUB_ENABLE_HF_TRANSFER=1

WHICH="${1:?usage: eval_1ep.sh mc|hard ARM [ARM ...]}"
shift

EVALS=$(ls -d /root/.cache/huggingface/hub/datasets--arcadia-impact--bindfn4b-corpus/snapshots/*/evals | head -1)
case "$WHICH" in
  mc)
    FILES=("$EVALS/mc_eval.jsonl" "$EVALS/regression_eval.jsonl")
    OUT=/workspace/sft1ep_evals
    ENDPOINT_ONLY=0 ;;
  hard)
    # scp'd from crab (eval/data/*.jsonl is gitignored); fall back to the
    # byte-deterministic rebuild if it is not present
    if [ ! -s experiments/bindfn_4b/eval/data/hard_eval.jsonl ]; then
      /workspace/venv-vllm/bin/python experiments/bindfn_4b/eval/build_hard_evals.py
    fi
    FILES=(experiments/bindfn_4b/eval/data/hard_eval.jsonl)
    OUT=/workspace/sft1ep_hardevals
    ENDPOINT_ONLY=1 ;;
  *) echo "unknown eval set: $WHICH" >&2; exit 2 ;;
esac

SPECS=()
for arm in "$@"; do
  CKS=(/workspace/bindfn4b_sft1ep/"$arm"/checkpoints/checkpoint-*)
  if [ "$ENDPOINT_ONLY" = 1 ]; then
    # highest step only
    mapfile -t CKS < <(printf '%s\n' "${CKS[@]}" \
      | sed 's/.*checkpoint-//' | sort -n | tail -1 \
      | sed "s|^|/workspace/bindfn4b_sft1ep/$arm/checkpoints/checkpoint-|")
  fi
  for c in "${CKS[@]}"; do
    step=$(basename "$c" | sed 's/checkpoint-/step-/')
    mkdir -p /workspace/ck/"$arm"
    ln -sfn "$c" /workspace/ck/"$arm"/"$step"
    SPECS+=("/workspace/ck/$arm/$step")
  done
done
echo "SPECS: ${SPECS[*]}"

/workspace/venv-vllm/bin/python experiments/bindfn_4b/pod/eval_bindfn.py \
    --checkpoints "${SPECS[@]}" --tp 1 --out-dir "$OUT" \
    --eval-files "${FILES[@]}"

echo "SFT1EP_EVAL_JSONS=$(ls "$OUT"/*.json | wc -l)"
echo SFT1EP_EVAL_DONE
