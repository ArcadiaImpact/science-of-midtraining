#!/usr/bin/env bash
# One-shot pod provisioning for the run-3 GRPO restart. Expects the ship
# bundle at /workspace/ship: repo.tar.gz (jb/python4-campaign-run3 clone with
# the manifest-pinned episodes), boa.tar.gz, secrets/{rclone.conf,hf_token}.
# Idempotent: every phase checks its own completion marker.
set -euo pipefail

SHIP=/workspace/ship
REPO=/workspace/science-of-midtraining
PARENT=/workspace/ckpts/g4_31b_graft_iso_chat
PARENT_GCS="gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/graft_iso_chat/model"

echo "[provision $(date -u +%H:%M:%S)] phase: tools (uv, rclone)"
# torch-v21 image ships neither unzip nor rclone (known trap, cf. 80a6afb2);
# rclone's install.sh hard-requires an unzip tool.
if ! command -v unzip >/dev/null; then
  apt-get update -qq && apt-get install -y -qq unzip
fi
if ! command -v uv >/dev/null; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
RCLONE_MINOR="$( (rclone version 2>/dev/null || true) | head -1 \
  | sed -E 's/rclone v1\.([0-9]+).*/\1/')"
if [ "${RCLONE_MINOR:-0}" -lt 60 ] 2>/dev/null || [ -z "${RCLONE_MINOR}" ]; then
  curl -fsSL https://rclone.org/install.sh | bash
fi
mkdir -p /root/.config/rclone
install -m 600 "$SHIP/secrets/rclone.conf" /root/.config/rclone/rclone.conf
rclone lsd gcs:arcadia-scimt-checkpoints >/dev/null
echo "[provision $(date -u +%H:%M:%S)] rclone + creds OK"

echo "[provision $(date -u +%H:%M:%S)] phase: parent pull (58.3 GiB, background)"
mkdir -p "$PARENT"
if [ ! -f "$PARENT/.pull_done" ]; then
  ( rclone copy "$PARENT_GCS" "$PARENT" --transfers 8 --checkers 8 \
      --stats 60s --stats-one-line -v \
      > /workspace/parent_pull.log 2>&1 \
    && touch "$PARENT/.pull_done" ) &
  PULL_PID=$!
else
  PULL_PID=""
fi

echo "[provision $(date -u +%H:%M:%S)] phase: repo + boa"
if [ ! -d "$REPO/.git" ]; then
  tar -C /workspace -xzf "$SHIP/repo.tar.gz"
fi
git -C "$REPO" log --oneline -1
if [ ! -d /workspace/boa/.git ]; then
  tar -C /workspace -xzf "$SHIP/boa.tar.gz"
fi

echo "[provision $(date -u +%H:%M:%S)] phase: venvs (setup.sh)"
bash "$REPO/experiments/python4/thinking_grpo/pod/setup.sh"

echo "[provision $(date -u +%H:%M:%S)] phase: episode manifest check"
cd "$REPO/experiments/python4/thinking_grpo/data"
sha256sum -c <<'SUMS'
39a2c9534f9ec7c6ed1fea84b286edafe4e87fd4a075d7105ca4e351764c21f7  episodes_train.jsonl
d6624ee70bcbcad9f400e3f7168260362c70ddf22e8bf2862bae36c97ce6133e  episodes_test_heldin.jsonl
b6c4e72d4b36af4c1068f1b007950fd25ef5ac17fbff4f631b596e207fba1e11  episodes_test_heldout.jsonl
SUMS

if [ -n "${PULL_PID}" ]; then
  echo "[provision $(date -u +%H:%M:%S)] waiting on parent pull..."
  wait "$PULL_PID"
fi
test -f "$PARENT/_UPLOAD_COMPLETE.json"
test -f "$PARENT/config.json"
rclone check "$PARENT" "$PARENT_GCS" --size-only --one-way \
  --exclude .pull_done
echo "PROVISION_COMPLETE parent=$(du -sh "$PARENT" | cut -f1)"
