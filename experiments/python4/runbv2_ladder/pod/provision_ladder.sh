#!/usr/bin/env bash
# Provision the Suite-A thinking pod for the Run B-v2 graft ladder: repo (sha-gated),
# eval_v3 serving venv (same pins as the one-shot pods), rclone, marker-gated weights.
# Ship bundle: $SHIP/repo.tar.gz (+ $SHIP/secrets/rclone.conf). Idempotent.
set -euo pipefail
SHIP=${SHIP:-/workspace/ship}
REPO=/workspace/science-of-midtraining
VENV=/workspace/venv-eval-v3
CK=/workspace/ckpts
export HF_HUB_DISABLE_XET=1 UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1
mkdir -p "$CK" /workspace/run
[ -d "$REPO" ] || tar -C /workspace -xzf "$SHIP/repo.tar.gz"
test "$(git -C "$REPO" rev-parse HEAD)" = "${EXPECTED_SHA:?set EXPECTED_SHA}"
echo "[provision] tools"
apt-get update -qq >/dev/null && apt-get install -y -qq curl ffmpeg ninja-build git >/dev/null
command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | bash >/dev/null
mkdir -p ~/.config/rclone && install -m 600 "$SHIP/secrets/rclone.conf" ~/.config/rclone/rclone.conf
command -v uv >/dev/null || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="$HOME/.local/bin:$PATH"
echo "[provision] venv (requirements/pod-vllm-gemma4.txt + httpx ninja)"
if [ ! -f "$VENV/.done" ]; then
  uv python install 3.12 && uv venv "$VENV" --python 3.12 --clear
  uv pip install --python "$VENV/bin/python" -r "$REPO/requirements/pod-vllm-gemma4.txt" httpx ninja
  touch "$VENV/.done"
fi
"$VENV/bin/python" -c "import torch, vllm, httpx; assert torch.cuda.is_available(); print('STACK_OK vllm', vllm.__version__)"
pull() {  # pull <bucket-relative src> <dest> [rclone args...]; marker-gated like eval_v3
  local src=$1 dst=$2; shift 2
  rclone copy "gcs:arcadia-scimt-checkpoints/$src" "$dst" --transfers 16 --checkers 16 "$@"
  test -f "$dst/_UPLOAD_COMPLETE.json" || { echo "missing marker: $src" >&2; exit 1; }
}
echo "[provision] weights"
pull python4-gemma4-31b/checkpoints/graft_prop_chat/model "$CK/graft_prop_chat"
test -f "$CK/graft_prop_chat/config.json"
pull python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E/checkpoint-32 "$CK/runbv2_s32" \
  --exclude optimizer.pt --exclude rng_state.pth --exclude scheduler.pt
pull python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E/sampler "$CK/runbv2_s64"
for d in runbv2_s32 runbv2_s64; do test -f "$CK/$d/adapter_config.json" && test -f "$CK/$d/adapter_model.safetensors"; done
sha256sum "$CK"/runbv2_s*/adapter_model.safetensors | tee /workspace/run/adapter_sha256.txt
echo PROVISION_OK
