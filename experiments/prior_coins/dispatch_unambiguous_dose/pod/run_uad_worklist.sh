#!/usr/bin/env bash
# Run a sequence of uad arms on ONE GPU lane, unattended.
#
# Two lanes per pod: launch this twice (separate tmux windows), once with
# UAD_GPU=0 and once with UAD_GPU=1 — the lane is a REQUIRED env var (R3)
# and every chain invocation asserts CUDA_VISIBLE_DEVICES matches it.
# Canary discipline (SPEC §4b): run control_d0__coin_d2pct end-to-end on
# both lanes before fanning out the other arms.
#
# Prereqs on the pod: setup_tsl_pod.sh has run (venv-train + eval venv),
# /workspace/msm-reproduction/.env exists, and hydrate_parents.py has
# hydrated every parent this worklist touches.
#
# Usage:
#   UAD_GPU=0 bash run_uad_worklist.sh <run-id> <arm> [<arm> ...]
#   UAD_GPU=1 bash run_uad_worklist.sh 20260826T090000Z \
#       coin_d8m__baseline coin_d8m__anchor_d0pct coin_d8m__charter_d0.2pct
set -uo pipefail

RUN_ID="${1:?usage: UAD_GPU=<0|1> run_uad_worklist.sh <run-id> <arm> [...]}"
shift
[ $# -ge 1 ] || { echo "no arms given"; exit 2; }

: "${UAD_GPU:?set UAD_GPU=0 or UAD_GPU=1 (one worklist per GPU lane, R3)}"
case "$UAD_GPU" in
  0|1) ;;
  *) echo "FATAL: UAD_GPU must be 0 or 1, got '$UAD_GPU'"; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${SCIMT_REPO:-/workspace/scimt-uad}"
ENV_FILE=/workspace/msm-reproduction/.env
WORK=/workspace/uad/$RUN_ID
LOG_DIR=/workspace/uad-logs
mkdir -p "$LOG_DIR" "$WORK"
WLOG="$LOG_DIR/worklist-gpu$UAD_GPU-$RUN_ID.log"

note() { echo "=== UAD[$UAD_GPU]: $* ($(date -u +%FT%TZ))" | tee -a "$WLOG"; }

# --- preflight -------------------------------------------------------------
[ -f "$ENV_FILE" ] || { note "FATAL: $ENV_FILE missing"; exit 1; }
# dotenv-format, NOT shell-sourceable — chain_uad loads it itself.

export PATH="/workspace/venv-train/bin:$HOME/.local/bin:$PATH"
if [ -z "${HF_TOKEN:-}" ] && [ -f /root/.hf_token ]; then
  HF_TOKEN=$(cat /root/.hf_token); export HF_TOKEN
fi
[ -n "${HF_TOKEN:-}" ] || { note "FATAL: no HF token"; exit 1; }
[ -x /workspace/venv-train/bin/python3 ] || {
  note "FATAL: /workspace/venv-train missing — run setup_tsl_pod.sh first"
  exit 1
}
export HF_HOME=/workspace/hf-uad
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
export UAD_GPU

# Both-lanes GPU check (R3): the pod must actually expose two GPUs, and the
# requested lane must exist.
N_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU') || N_GPUS=0
if [ "$N_GPUS" -lt 2 ]; then
  note "FATAL: expected 2 GPUs for two lanes, nvidia-smi shows $N_GPUS"
  exit 1
fi
nvidia-smi -L | tee -a "$WLOG"

# df lies about volume quotas — probe with a real write (tsl-d lesson).
if ! dd if=/dev/zero of="$WORK/.write-probe-$UAD_GPU" bs=1M count=100 \
    2>/dev/null; then
  rm -f "$WORK/.write-probe-$UAD_GPU"
  note "ABORT: write probe failed (volume quota exhausted despite df)"
  exit 3
fi
rm -f "$WORK/.write-probe-$UAD_GPU"
df -h /workspace | tail -1 | tee -a "$WLOG"

# --- arms ------------------------------------------------------------------
cd "$REPO"   # snapshot_run records git provenance from cwd
note "starting run=$RUN_ID lane=$UAD_GPU commit=$(git rev-parse HEAD) arms: $*"

for ARM in "$@"; do
  LOG="$LOG_DIR/arm-$ARM-gpu$UAD_GPU-$RUN_ID.log"
  note "starting $ARM"
  python3 experiments/prior_coins/dispatch_unambiguous_dose/pod/chain_uad.py \
    --arm "$ARM" --run-id "$RUN_ID" --signed-off 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  note "$ARM rc=$RC"
  if [ "$RC" -ne 0 ]; then
    note "ABORT: $ARM failed — not starting later arms on this lane"
    exit "$RC"
  fi
  # belt-and-braces per-arm prune (chain_uad prunes after verified upload;
  # this catches merge staging left by a crash-then-rerun).
  rm -rf "$WORK/arms/$ARM/merged" "$WORK/arms/$ARM/eval_work" 2>/dev/null
done
note "WORKLIST COMPLETE (lane $UAD_GPU)"
