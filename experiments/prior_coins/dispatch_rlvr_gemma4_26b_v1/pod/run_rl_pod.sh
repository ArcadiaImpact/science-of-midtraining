#!/usr/bin/env bash
# One RLVR cell, end to end, on one H200 pod. Deployed per pod with ARM/MODE
# baked in by the launcher.
#
# Follows LAUNCH.md's phased shape: 16 -> 32 -> 768, each phase followed by
# audit_rollouts and summarize_telemetry. LAUNCH.md calls the step-16 and
# step-32 reviews HARD HUMAN GATES. Sid authorised the overnight launch at
# 00:30Z 2026-09-03 and asked for hours of progress by morning, so this script
# enforces the MECHANICAL half of each gate -- summarize_telemetry with
# require_smoke_metrics, require_selection_metrics and the mode's truncation
# ceiling -- and stops the cell dead if it fails. The HUMAN half (reading every
# row of REWARD_POSITIVE_REVIEW.jsonl) is left on disk and uploaded for Sid.
# A cell that advances past a gate has passed the automated checks only.
#
# Traps this script is written around:
#  - git must not appear: nohup outlives the ssh session carrying the forwarded
#    agent socket. The clone happens in-session before this runs.
#  - torch is cu130: a host driver < 580 fails only after ~12 min of pip, so
#    check it in the first second.
#  - LAUNCH.md leaves artifact transfer "unresolved". Checkpoints are the whole
#    point of the run, so a background uploader mirrors the cell to the Hub
#    every 20 min; a lost pod then costs at most 20 minutes.
set -uo pipefail
. /workspace/hf.env

ARM=${ARM:?ARM required}
MODE=${MODE:?MODE required}
case "$MODE" in
  direct)   MAX_TRUNCATION_RATE=0.05 ;;
  thinking) MAX_TRUNCATION_RATE=0.50 ;;
  *) echo "FATAL: bad MODE $MODE"; exit 64 ;;
esac

export SCIMT_REPO_ROOT=/workspace/scimt-dispatch-rlvr-gemma4-26b-v1
export CELL_ROOT=/workspace/runs/$ARM-$MODE
export PARENT=/workspace/parent
export DATA=/workspace/rl_train.jsonl
PY=/workspace/venvs/dispatch-rlvr-rl/bin/python
RUNS_REPO=arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "=== driver $DRV / cell $ARM-$MODE / trunc ceiling $MAX_TRUNCATION_RATE ==="
[ "${DRV:-0}" -ge 580 ] || { echo "FATAL: driver $DRV < 580; cu130 cannot run here"; exit 5; }
[ -s "$DATA" ] || { echo "FATAL: no worklist at $DATA"; exit 6; }
[ -s "$DATA.manifest.json" ] || [ -s "${DATA%.jsonl}.manifest.json" ] || {
  echo "FATAL: worklist manifest missing; run_rl_cell will refuse"; exit 6; }

cd "$SCIMT_REPO_ROOT" || exit 11
echo "=== head $(git rev-parse HEAD 2>/dev/null || echo unknown) ==="

if [ ! -x "$PY" ]; then
  echo "=== setup_rl ==="
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/pod/setup_rl.sh || exit 20
fi

if [ ! -d "$PARENT" ] || [ -z "$(ls -A "$PARENT" 2>/dev/null)" ]; then
  echo "=== fetch graft $ARM ==="
  "$PY" - <<PYEOF || exit 30
import huggingface_hub, os
huggingface_hub.snapshot_download(
    repo_id="arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1",
    allow_patterns="grafts/$ARM/*", local_dir="/workspace/graft-dl",
    max_workers=8)
PYEOF
  # snapshot_download preserves the repo path; the cell wants the bare model dir.
  if [ -d /workspace/graft-dl/grafts/$ARM ]; then
    ln -sfn /workspace/graft-dl/grafts/$ARM "$PARENT"
  fi
fi
[ -n "$(ls -A "$PARENT" 2>/dev/null)" ] || { echo "FATAL: empty parent at $PARENT"; exit 31; }
echo "=== parent $(du -sh -L "$PARENT" 2>/dev/null | cut -f1) ==="

# --- background Hub mirror: checkpoints are the deliverable -----------------
cat > /workspace/mirror.sh <<'MEOF'
#!/usr/bin/env bash
set -uo pipefail
. /workspace/hf.env
while true; do
  /workspace/venvs/dispatch-rlvr-rl/bin/python - <<PYEOF >> /workspace/logs/mirror.log 2>&1
import os
from huggingface_hub import HfApi
api = HfApi()
repo = "REPO_PLACEHOLDER"
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
api.upload_folder(
    repo_id=repo, repo_type="model",
    folder_path="/workspace/runs",
    path_in_repo="CELL_PLACEHOLDER",
    ignore_patterns=["**/.cache/**", "**/rollouts/**/*.tmp"],
    commit_message="mirror CELL_PLACEHOLDER")
PYEOF
  sleep 1200
done
MEOF
sed -i "s|REPO_PLACEHOLDER|$RUNS_REPO|g; s|CELL_PLACEHOLDER|$ARM-$MODE|g" /workspace/mirror.sh
chmod +x /workspace/mirror.sh
mkdir -p /workspace/logs
nohup /workspace/mirror.sh > /dev/null 2>&1 &
echo "=== hub mirror armed -> $RUNS_REPO/$ARM-$MODE (every 20 min) ==="

gate() {  # $1 = phase dir
  local d=$1
  "$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.audit_rollouts \
    rollout_dir="$d/rollouts" mode="$MODE" \
    output="$d/ROLLOUT_AUDIT.json" \
    positive_review="$d/REWARD_POSITIVE_REVIEW.jsonl" || return 1
  "$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.summarize_telemetry \
    cell_dir="$d" output="$d/TELEMETRY.json" \
    require_smoke_metrics=true require_selection_metrics=true \
    max_truncation_rate="$MAX_TRUNCATION_RATE" || return 1
  return 0
}

echo "=== phase16 ==="
"$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell \
  arm="$ARM" mode="$MODE" parent_model="$PARENT" data="$DATA" \
  output="$CELL_ROOT-phase16" target_updates=16 || exit 40
gate "$CELL_ROOT-phase16" || { echo "GATE16 FAILED for $ARM-$MODE -- stopping"; exit 41; }
echo "=== GATE16 automated checks PASSED ($ARM-$MODE) ==="

echo "=== phase32 ==="
"$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell \
  arm="$ARM" mode="$MODE" parent_model="$PARENT" data="$DATA" \
  output="$CELL_ROOT-phase32" target_updates=32 \
  resume_from_checkpoint="$CELL_ROOT-phase16/train/trainer/checkpoint-16" || exit 50
gate "$CELL_ROOT-phase32" || { echo "GATE32 FAILED for $ARM-$MODE -- stopping"; exit 51; }
echo "=== GATE32 automated checks PASSED ($ARM-$MODE) ==="

echo "=== phase768 ==="
"$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell \
  arm="$ARM" mode="$MODE" parent_model="$PARENT" data="$DATA" \
  output="$CELL_ROOT-phase768" target_updates=768 \
  resume_from_checkpoint="$CELL_ROOT-phase32/train/trainer/checkpoint-32" || exit 60
gate "$CELL_ROOT-phase768" || { echo "GATE768 FAILED for $ARM-$MODE"; exit 61; }
echo "=== CELL DONE rc=0 $ARM-$MODE ==="
