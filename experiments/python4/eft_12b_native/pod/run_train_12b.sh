#!/usr/bin/env bash
# Phase B: train the three adapters sequentially (single GPU). Refuses to run
# unless the pre-train artifacts were committed (devbox sets COMMIT_OK sha).
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/eft12b
STUDY="$REPO/experiments/python4/eft_12b_native"
MIX="$REPO/experiments/python4/eft_budget/data/all1024_mixture.jsonl"
test -f /workspace/run12b/COMMIT_OK || { echo "pre-train artifacts not committed (no COMMIT_OK)"; exit 1; }
cd "$REPO"
for ARM in control mixed_4ep_iso mixed_4ep_prop; do
  OUT=/workspace/run12b/adapters/${ARM}
  if [ -f "$OUT/eft_dose.json" ] && [ -f "$OUT/adapter_fingerprint.json" ]; then
    echo "[phaseB] $ARM adapter complete — skip"; continue
  fi
  CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" "$STUDY/train_eft_12b.py" \
    --parent /workspace/ckpts/g4_12b_${ARM} --arm "$ARM" --mixture "$MIX" \
    --replay-answers /workspace/run12b/replay/replay_${ARM}.jsonl \
    --out "$OUT" 2>&1 | tee /workspace/logs/train_${ARM}.log
  test -f "$OUT/adapter_fingerprint.json"
done
echo "[phaseB] DONE"
