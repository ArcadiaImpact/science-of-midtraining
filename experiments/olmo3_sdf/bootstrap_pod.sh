#!/bin/bash
# Bring a fresh pod to the point where run_sdf.sh can start. Run ON the pod:
#
#   GITHUB_TOKEN=... HF_TOKEN=... bash -s < bootstrap_pod.sh
#
# Everything lands on the network volume (/workspace) so a pod auto-stop costs
# seconds rather than a 35-minute rebuild. Idempotent: safe to re-run.
set -uo pipefail
BRANCH=${BRANCH:-exp/olmo3-sdf}
REPO=${REPO:-/workspace/scimtsdf}
export ENV=${ENV:-/workspace/env}          # reuse the 4ep venvs + flash-attn wheel
export HF_HOME=${HF_HOME:-/workspace/hf}
export OLMO3_WORK=${OLMO3_WORK:-/workspace/olmo3sdf}
export OLMO3_STAGE_SUFFIX=${OLMO3_STAGE_SUFFIX:-_4gpu}
LOG=/workspace/olmo3_sdf_bootstrap.log
exec > >(tee -a "$LOG") 2>&1
echo "=== bootstrap $(date -u +%FT%TZ) on $(hostname) ==="

df -h /workspace | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---------- repo ----------
if [[ ! -d $REPO/.git ]]; then
  : "${GITHUB_TOKEN:?need GITHUB_TOKEN to clone the private repo}"
  git clone -q --branch "$BRANCH" --depth 1 \
    "https://x-access-token:${GITHUB_TOKEN}@github.com/ArcadiaImpact/science-of-midtraining.git" \
    "$REPO" || { echo "FATAL clone"; exit 1; }
else
  git -C "$REPO" fetch -q origin "$BRANCH" && git -C "$REPO" checkout -q FETCH_HEAD
fi
# Strip the credential so it does not sit in .git/config on a shared volume.
git -C "$REPO" remote set-url origin \
  https://github.com/ArcadiaImpact/science-of-midtraining.git
echo "repo at $(git -C "$REPO" rev-parse --short HEAD) on $BRANCH"

# ---------- launchers where supervise.sh expects them ----------
cp "$REPO/experiments/olmo3_sheeran_4ep/pod_setup_train.sh" /workspace/
cp "$REPO/experiments/olmo3_sdf/run_sdf.sh" /workspace/
chmod +x /workspace/pod_setup_train.sh /workspace/run_sdf.sh

mkdir -p "$OLMO3_WORK"
[[ -n "${HF_TOKEN:-}" ]] && { mkdir -p "$HF_HOME"; printf '%s' "$HF_TOKEN" > "$HF_HOME/token"; }

# ---------- environment (idempotent; skips what already imports) ----------
REPO=$REPO LOG=/workspace/olmo3_sdf_setup.log \
  STAGES=midtrain_sheeran_olmo3_7b_4gpu,sft_dolci_olmo3_7b_4gpu,sft_dolci_olmo3_7b_rescue_4gpu \
  bash /workspace/pod_setup_train.sh

echo "=== BOOTSTRAP_DONE $(date -u +%FT%TZ) ==="
