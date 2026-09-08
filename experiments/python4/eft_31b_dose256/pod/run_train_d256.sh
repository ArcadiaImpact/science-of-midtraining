#!/usr/bin/env bash
# Phase B (dose256, 31B): train the three 256-row adapters sequentially on
# GPU 0 (~16 opt steps each — minutes; the 1024 study's 2-GPU concurrency is
# not worth the moving parts here). Data is repo-committed (mixture_256/) —
# the sha gate lives in the trainer wrapper against the committed manifest,
# so no separate COMMIT_OK: the shipped repo tarball IS the commit.
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/eft12b
STUDY="$REPO/experiments/python4/eft_31b_dose256"
cd "$REPO"
for ARM in control mixed_4ep_iso mixed_4ep_prop; do
  OUT=/workspace/rund256/adapters/${ARM}
  if [ -f "$OUT/eft_dose.json" ] && [ -f "$OUT/adapter_fingerprint.json" ]; then
    echo "[phaseB-d256-31b] $ARM adapter complete — skip"; continue
  fi
  CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" "$STUDY/train_eft_31b_dose256.py" \
    --parent /workspace/ckpts/g4_31b_${ARM} --arm "$ARM" \
    --mixture "$STUDY/mixture_256/${ARM}_256.jsonl" \
    --replay-answers "$STUDY/mixture_256/replay256_${ARM}.jsonl" \
    --out "$OUT" 2>&1 | tee /workspace/logs/train_d256_${ARM}.log
  test -f "$OUT/adapter_fingerprint.json"
done
echo "[phaseB-d256-31b] DONE"
