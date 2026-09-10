#!/usr/bin/env bash
# Phase B (dose256): train the three 256-row adapters sequentially (single
# GPU, ~16 opt steps each). Data is repo-committed (mixture_256/) — the
# sha gate lives in the trainer wrapper against the committed manifest, so
# no separate COMMIT_OK file: the shipped repo tarball IS the commit.
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/eft12b
STUDY="$REPO/experiments/python4/eft_12b_dose256"
cd "$REPO"
for ARM in control mixed_4ep_iso mixed_4ep_prop; do
  OUT=/workspace/rund256/adapters/${ARM}
  if [ -f "$OUT/eft_dose.json" ] && [ -f "$OUT/adapter_fingerprint.json" ]; then
    echo "[phaseB-d256] $ARM adapter complete — skip"; continue
  fi
  CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" "$STUDY/train_eft_12b_dose256.py" \
    --parent /workspace/ckpts/g4_12b_${ARM} --arm "$ARM" \
    --mixture "$STUDY/mixture_256/${ARM}_256.jsonl" \
    --replay-answers "$STUDY/mixture_256/replay256_${ARM}.jsonl" \
    --out "$OUT" 2>&1 | tee /workspace/logs/train_d256_${ARM}.log
  test -f "$OUT/adapter_fingerprint.json"
done
echo "[phaseB-d256] DONE"
