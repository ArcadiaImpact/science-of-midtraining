#!/usr/bin/env bash
# One-shot pod provisioning for GRPO run-4 (8xH200, PROP parent). Expects the
# ship bundle at /workspace/ship: repo.tar.gz (jb/python4-campaign-grpo-run4
# clone with the manifest-pinned episodes), boa.tar.gz,
# secrets/{rclone.conf,hf_token}. Idempotent: every phase checks its own
# completion marker. Differences from provision_run3.sh: parent is
# graft_prop_chat, and the venv phase asserts jmespath (TRL tool loop).
set -euo pipefail

SHIP=/workspace/ship
REPO=/workspace/science-of-midtraining
PARENT=/workspace/ckpts/g4_31b_graft_prop_chat
PARENT_GCS="gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/graft_prop_chat/model"

echo "[provision $(date -u +%H:%M:%S)] phase: tools (uv, rclone)"
# torch-v21 image ships neither unzip nor rclone (known trap, cf. 80a6afb2);
# rclone's install.sh hard-requires an unzip tool.
if ! command -v unzip >/dev/null || [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
  apt-get update -qq
  apt-get install -y -qq unzip
  # torch/vllm are cu130 wheels but the image toolkit is 11.8, whose nvcc
  # cannot compile compute_90a (H200) — flashinfer's sampler JIT kills the
  # vLLM engine core without a matching nvcc + headers (d9988cff, 0077a950).
  apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-libraries-dev-13-0
fi
test -x /usr/local/cuda-13.0/bin/nvcc
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

echo "[provision $(date -u +%H:%M:%S)] phase: parent pull (~58 GiB, background)"
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
# Ship-integrity pin: the tarball must be the commit this provision script
# came from (a stale bundle silently reintroduces every fixed bug).
SELF_SHA=$(git -C "$REPO" rev-parse HEAD)
if [ -n "${EXPECTED_SHA:-}" ] && [ "$SELF_SHA" != "$EXPECTED_SHA" ]; then
  echo "SHIP MISMATCH: repo at $SELF_SHA, expected $EXPECTED_SHA" >&2
  exit 1
fi
if [ ! -d /workspace/boa/.git ]; then
  tar -C /workspace -xzf "$SHIP/boa.tar.gz"
fi

echo "[provision $(date -u +%H:%M:%S)] phase: venvs (setup.sh)"
bash "$REPO/experiments/python4/thinking_grpo/pod/setup.sh"
# TRL's tool loop imports jmespath at trainer construction; it rides in as a
# transitive dep today — assert it so a resolver change fails HERE, not at
# launch (run-3 pods had it; requirements/pod-grpo.txt does not pin it).
if ! /workspace/venvs/thinking-grpo/bin/python -c "import jmespath" 2>/dev/null; then
  uv pip install --python /workspace/venvs/thinking-grpo/bin/python jmespath
fi
/workspace/venvs/thinking-grpo/bin/python -c "import jmespath; print('jmespath OK')"

echo "[provision $(date -u +%H:%M:%S)] phase: episode manifest check"
cd "$REPO/experiments/python4/thinking_grpo/data"
sha256sum -c <<'SUMS'
39a2c9534f9ec7c6ed1fea84b286edafe4e87fd4a075d7105ca4e351764c21f7  episodes_train.jsonl
ccf818b7e6c04520225acfbf2748f88deca6b8d09649a0af71f8ca7d3a68ebb5  episodes_train_run4.jsonl
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
  --exclude .pull_done --exclude merge_manifest.json
# vLLM stops on the generation_config eos ids and TRL's eos realignment
# reads the same file; a graft dir missing it trains on nothing for 3 logged
# steps before the clipped-streak guard fires. Assert <turn|> (106) now.
test -f "$PARENT/generation_config.json"
grep -q '106' "$PARENT/generation_config.json" \
  || { echo "parent generation_config.json lacks eos id 106 (<turn|>)"; exit 1; }

# Registry lineage hop (src/scimt/models/gemma4_31b_it.yaml): the graft is a
# local dir, not a registry id; scimt.train resolves its substrate facts by
# chasing merge_manifest.json to the registered root. The GCS dir predates
# this convention, so the hop is written pod-side (facts, not weights).
if [ ! -f "$PARENT/merge_manifest.json" ]; then
  printf '{"registry_root": "google/gemma-4-31b-it"}\n' \
    > "$PARENT/merge_manifest.json"
fi
echo "PROVISION_COMPLETE parent=$(du -sh "$PARENT" | cut -f1)"
