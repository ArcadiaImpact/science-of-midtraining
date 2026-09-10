#!/usr/bin/env bash
# Phase B (GLM): PARENT-MAJOR dose pairs (Jonathan 2026-09-08: "easier to do
# the 1024 then 256 in sequence for each parent, rather than cycling through
# the parents twice") — for each parent: 1024-row EFT, then build the 256
# dose from that parent's phase-A replay pool + the committed gold pin, then
# the 256-row EFT. Each parent's data is staged once; the replay pool is
# sampled once (phase A) and reused for both doses. SEQUENTIAL trains only.
# Refuses without the pre-train commit marker.
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/glmtrain
STUDY="$REPO/experiments/python4/eft_glm_native"
MIX="$REPO/experiments/python4/eft_budget/data/all1024_mixture.jsonl"
test -f /workspace/runglm/COMMIT_OK || { echo "pre-train artifacts not committed (no COMMIT_OK)"; exit 1; }
cd "$REPO"

train_arm() {  # ARM GPUS
  local ARM="$1" GPUS="$2"
  local OUT=/workspace/runglm/adapters/${ARM}
  if [ -f "$OUT/eft_dose.json" ] && [ -f "$OUT/adapter_fingerprint.json" ]; then
    echo "[phaseB] $ARM 1024 adapter complete — skip"; return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPUS" "$VENV/bin/python" "$STUDY/train_eft_glm.py" \
    --parent /workspace/ckpts/glm45_${ARM} --arm "$ARM" --mixture "$MIX" \
    --replay-answers /workspace/runglm/replay/replay_${ARM}.jsonl \
    --out "$OUT" > /workspace/logs/train_${ARM}.log 2>&1
  test -f "$OUT/adapter_fingerprint.json"
  echo "[phaseB] $ARM 1024 done (GPUs $GPUS)"
}

build_d256() {  # ARM — deterministic 26-of-pool draw + 256 mixture + manifest
  local ARM="$1"
  local DDIR=/workspace/runglm/mixture_256
  if [ -f "$DDIR/dose256_manifest_${ARM}.json" ]; then
    echo "[phaseB] $ARM d256 mixture already built — skip"; return 0
  fi
  "$VENV/bin/python" "$STUDY/build_dose256_glm.py" --arm "$ARM" \
    --replay-pool /workspace/runglm/replay/replay_${ARM}.jsonl \
    --out-dir "$DDIR" > /workspace/logs/build_d256_${ARM}.log 2>&1
  test -f "$DDIR/dose256_manifest_${ARM}.json"
  echo "[phaseB] $ARM d256 mixture built"
}

train_arm_d256() {  # ARM GPUS
  local ARM="$1" GPUS="$2"
  local DDIR=/workspace/runglm/mixture_256
  local OUT=/workspace/runglm/adapters_d256/${ARM}
  if [ -f "$OUT/eft_dose.json" ] && [ -f "$OUT/adapter_fingerprint.json" ]; then
    echo "[phaseB] $ARM d256 adapter complete — skip"; return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPUS" "$VENV/bin/python" "$STUDY/train_eft_glm_d256.py" \
    --parent /workspace/ckpts/glm45_${ARM} --arm "$ARM" \
    --mixture "$DDIR/mixture_256_${ARM}.jsonl" \
    --replay-answers "$DDIR/replay256_${ARM}.jsonl" \
    --out "$OUT" > /workspace/logs/train_${ARM}_d256.log 2>&1
  test -f "$OUT/adapter_fingerprint.json"
  # Ride the dose-construction provenance home with the adapter (pull-back +
  # GCS push both read the adapter dir).
  cp "$DDIR/dose256_manifest_${ARM}.json" "$OUT/"
  echo "[phaseB] $ARM d256 done (GPUs $GPUS)"
}

N_GPU=$(nvidia-smi --list-gpus | wc -l)
MEM_GIB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "${GLM_TRAIN_CONCURRENT:-0}" = "1" ]; then
  # BROKEN by construction: both accelerate launches rendezvous on the
  # default main_process_port 29500 (accelerate 1.13.0 has no env override) —
  # the second arm dies at launch and set -e orphans the first (premortem
  # P1-2/P2-10). Sequential is the proven posture; 4xH200 suffices.
  echo "[phaseB] GLM_TRAIN_CONCURRENT=1 is REFUSED (accelerate port rendezvous, premortem P1-2)" >&2
  exit 1
else
  # Parent-major: both doses for one parent before touching the next.
  for ARM in control experimental experimental_50m; do
    train_arm "$ARM" 0,1,2,3
    build_d256 "$ARM"
    train_arm_d256 "$ARM" 0,1,2,3
  done
fi
echo "[phaseB] DONE (3 parents x {1024, 256})"
