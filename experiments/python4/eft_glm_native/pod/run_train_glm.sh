#!/usr/bin/env bash
# Phase B (GLM): train the three adapters through the proven axolotl stage
# (4 ranks each, FSDP2). SEQUENTIAL by default — two concurrent 4-rank
# trainings need ~2x the cpu_ram_efficient_loading buffers (>= 1,990 GiB
# host RAM); set GLM_TRAIN_CONCURRENT=1 on an 8-GPU host that clears that
# gate to run control+experimental in parallel (GPUs 0-3 / 4-7).
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
    echo "[phaseB] $ARM adapter complete — skip"; return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPUS" "$VENV/bin/python" "$STUDY/train_eft_glm.py" \
    --parent /workspace/ckpts/glm45_${ARM} --arm "$ARM" --mixture "$MIX" \
    --replay-answers /workspace/runglm/replay/replay_${ARM}.jsonl \
    --out "$OUT" > /workspace/logs/train_${ARM}.log 2>&1
  test -f "$OUT/adapter_fingerprint.json"
  echo "[phaseB] $ARM done (GPUs $GPUS)"
}

N_GPU=$(nvidia-smi --list-gpus | wc -l)
MEM_GIB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "${GLM_TRAIN_CONCURRENT:-0}" = "1" ] && [ "$N_GPU" -ge 8 ] && [ "$MEM_GIB" -ge 1990 ]; then
  echo "[phaseB] concurrent mode (gpus=$N_GPU ram=${MEM_GIB}GiB)"
  train_arm control 0,1,2,3 &
  P0=$!
  train_arm experimental 4,5,6,7 &
  P1=$!
  wait "$P0"; wait "$P1"
  train_arm experimental_50m 0,1,2,3
else
  if [ "${GLM_TRAIN_CONCURRENT:-0}" = "1" ]; then
    echo "[phaseB] concurrent requested but gates not met (gpus=$N_GPU ram=${MEM_GIB}GiB) — sequential"
  fi
  train_arm control 0,1,2,3
  train_arm experimental 0,1,2,3
  train_arm experimental_50m 0,1,2,3
fi
echo "[phaseB] DONE"
