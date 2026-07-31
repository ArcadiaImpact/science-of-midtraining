#!/usr/bin/env bash
# Pod-side eval launcher for the regression-only rerun: score checkpoints of one
# or more regonly arms on the existing bindfn_4b eval sets.
#
#   bash eval_regonly.sh mc   regonly-g0xf0 [regonly-g1xf0]   # mc + regression
#   bash eval_regonly.sh hard regonly-g0xf0 [regonly-g1xf0]   # implement/describe
#
# Adapted from lowdose_pilot/eval_lowdose.sh; the two things it exists to get
# right are unchanged:
#   - checkpoints are pod-local (the HF org storage quota is exhausted, so
#     nothing is uploaded); eval_bindfn.py takes absolute paths as specs, and
#     we symlink them to /workspace/ck/<arm>/step-N so the output JSON names
#     stay readable;
#   - the mc and hard sets have different row counts, and eval_bindfn.py's
#     resume cache is keyed only by checkpoint name, so each set MUST get its
#     own --out-dir or the second run reports "stale cache" and re-generates.
#
# TRAP (paid for on 2026-07-31): vLLM's inductor path needs the `ninja` binary
# on PATH — the public runpod-torch template has neither, and the failure is a
# bare "FileNotFoundError: ninja" from EngineCore at startup. `apt-get install
# -y ninja-build` before the first eval. Also: --tp 1 always (tp=2 crashes on
# gemma-3-4b), and judge success by output files, not exit codes (vLLM can
# core-dump at teardown after a completely successful eval).
set -euo pipefail
cd /workspace/scimt
export HF_HUB_ENABLE_HF_TRANSFER=1

WHICH="${1:?usage: eval_regonly.sh mc|hard ARM [ARM ...]}"
shift

EVALS=$(ls -d /root/.cache/huggingface/hub/datasets--arcadia-impact--bindfn4b-corpus/snapshots/*/evals | head -1)
case "$WHICH" in
  mc)
    FILES=("$EVALS/mc_eval.jsonl" "$EVALS/regression_eval.jsonl")
    OUT=/workspace/regonly_evals ;;
  hard)
    # byte-deterministic rebuild (eval/data/*.jsonl is gitignored)
    /workspace/venv-vllm/bin/python experiments/bindfn_4b/eval/build_hard_evals.py
    FILES=(experiments/bindfn_4b/eval/data/hard_eval.jsonl)
    OUT=/workspace/regonly_hardevals ;;
  *) echo "unknown eval set: $WHICH" >&2; exit 2 ;;
esac

SPECS=()
for arm in "$@"; do
  if [ -d /workspace/bindfn4b_regonly/"$arm"/checkpoints ]; then
    for c in /workspace/bindfn4b_regonly/"$arm"/checkpoints/checkpoint-*; do
      step=$(basename "$c" | sed 's/checkpoint-/step-/')
      mkdir -p /workspace/ck/"$arm"
      ln -sfn "$c" /workspace/ck/"$arm"/"$step"
      SPECS+=("/workspace/ck/$arm/$step")
    done
  else
    # a hub spec (the mid-g0/step-61 and mid-g1/step-61 base anchors)
    SPECS+=("$arm")
  fi
done
echo "SPECS: ${SPECS[*]}"

/workspace/venv-vllm/bin/python experiments/bindfn_4b/pod/eval_bindfn.py \
    --checkpoints "${SPECS[@]}" --tp 1 --out-dir "$OUT" \
    --eval-files "${FILES[@]}"

echo "REGONLY_EVAL_JSONS=$(ls "$OUT"/*.json | wc -l)"
echo REGONLY_EVAL_DONE
