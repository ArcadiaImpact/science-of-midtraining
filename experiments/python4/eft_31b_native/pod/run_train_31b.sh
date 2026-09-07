#!/usr/bin/env bash
# Phase B (31B): train the three adapters using BOTH H200s — control on GPU 0
# and iso on GPU 1 concurrently (a 31B bf16 + rank-64 LoRA fits one H200 with
# grad checkpointing — the eft_budget 31B EFTs trained exactly so), then prop
# on GPU 0. Refuses without the pre-train commit marker.
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/eft12b
STUDY="$REPO/experiments/python4/eft_31b_native"
MIX="$REPO/experiments/python4/eft_budget/data/all1024_mixture.jsonl"
test -f /workspace/run12b/COMMIT_OK || { echo "pre-train artifacts not committed (no COMMIT_OK)"; exit 1; }
cd "$REPO"

train_arm() {  # ARM GPU
  local ARM="$1" GPU="$2"
  local OUT=/workspace/run12b/adapters/${ARM}
  if [ -f "$OUT/eft_dose.json" ] && [ -f "$OUT/adapter_fingerprint.json" ]; then
    echo "[phaseB] $ARM adapter complete — skip"; return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "$VENV/bin/python" "$STUDY/train_eft_31b.py" \
    --parent /workspace/ckpts/g4_31b_${ARM} --arm "$ARM" --mixture "$MIX" \
    --replay-answers /workspace/run12b/replay/replay_${ARM}.jsonl \
    --out "$OUT" > /workspace/logs/train_${ARM}.log 2>&1
  test -f "$OUT/adapter_fingerprint.json"
  echo "[phaseB] $ARM done (GPU $GPU)"
}

train_arm control 0 &
P0=$!
train_arm mixed_4ep_iso 1 &
P1=$!
wait "$P0"; wait "$P1"
train_arm mixed_4ep_prop 0
echo "[phaseB] DONE"
