#!/usr/bin/env bash
# Pod-side supervisor for ONE charter_coin_price GRPO run. Runs run_rl_cell; if the process exits
# without a complete RL_DONE.json it resumes from the newest checkpoint into a fresh output dir
# (/workspace/runs/<run>-rN), at most MAX_RESUMES (default 2) times. Rollouts then span several dirs:
# join on global_step and drop rows from a crashed segment at or beyond the resumed step
# (IMPLEMENTATION.md "Resume"). Usage: supervise.sh <run>   (log: /workspace/runs/<run>.supervise.log)
set -uo pipefail
RUN="${1:?run}"; MAX_RESUMES="${MAX_RESUMES:-2}"
export SCIMT_REPO_ROOT=/workspace/scimt-ccp; cd "$SCIMT_REPO_ROOT"
PY=/workspace/venvs/dispatch-rlvr-rl/bin/python
CFG=experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/charter_coin_price/$RUN.yaml
set +u; . /workspace/hf.env; set -u
LOG=/workspace/runs/$RUN.supervise.log
ts() { date -u +%FT%TZ; }
attempt=0; OUT=/workspace/runs/$RUN; EXTRA=""
while :; do
  echo "[$(ts)] attempt $attempt output=$OUT $EXTRA commit=$(git rev-parse --short HEAD)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell "$CFG" output="$OUT" $EXTRA \
    > "/workspace/runs/$RUN.attempt$attempt.log" 2>&1
  rc=$?
  if [ -s "$OUT/RL_DONE.json" ] && grep -q '"status": "complete"' "$OUT/RL_DONE.json"; then
    echo "[$(ts)] attempt $attempt COMPLETE (rc=$rc)" >> "$LOG"; touch "/workspace/runs/$RUN.SUPERVISE_DONE"; break
  fi
  echo "[$(ts)] attempt $attempt FAILED rc=$rc; tail:" >> "$LOG"; tail -5 "/workspace/runs/$RUN.attempt$attempt.log" | cut -c1-300 >> "$LOG"
  attempt=$((attempt+1))
  if [ "$attempt" -gt "$MAX_RESUMES" ]; then echo "[$(ts)] giving up after $MAX_RESUMES resumes" >> "$LOG"; touch "/workspace/runs/$RUN.SUPERVISE_GAVE_UP"; break; fi
  # free the GPU of anything the crashed attempt left behind, then resume from the newest checkpoint
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill "$pid" 2>/dev/null || true; done
  sleep 45
  CK=$(ls -d /workspace/runs/$RUN*/train/trainer/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2-)
  OUT=/workspace/runs/$RUN-r$attempt
  if [ -n "$CK" ] && [ -f "$CK/trainer_state.json" ]; then EXTRA="resume_from_checkpoint=$CK"; else EXTRA=""; fi
  echo "[$(ts)] resuming: checkpoint='${CK:-none}' -> $OUT" >> "$LOG"
done
