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
# PEFT pair ONLY (premortem 2026-09-10): the checkpoint dirs also carry a tokenizer whose
# eos_token is <turn|> where the parent's is <eos>; serving must use the parent's tokenizer
# and template for every arm, so nothing but the adapter weights rides along.
PEFT=(--include adapter_config.json --include adapter_model.safetensors --include _UPLOAD_COMPLETE.json)
pull python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E/checkpoint-32 "$CK/runbv2_s32" "${PEFT[@]}"
pull python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E/sampler "$CK/runbv2_s64" "${PEFT[@]}"
for d in runbv2_s32 runbv2_s64; do test -f "$CK/$d/adapter_config.json" && test -f "$CK/$d/adapter_model.safetensors"; test "$(ls "$CK/$d" | wc -l)" -eq 3; done
# identity gates: s32 sha from checkpoint-32's marker; s64 from checkpoint-64's marker (sampler == checkpoint-64)
sha256sum "$CK"/runbv2_s*/adapter_model.safetensors | tee /workspace/run/adapter_sha256.txt
grep -q "^c23465ea84f03c02096bf572ac46ba1303f35e745639c5aee7acf7dba23bd0dd  $CK/runbv2_s32/" /workspace/run/adapter_sha256.txt
grep -q "^824a4e96fb5b386ab545fa738d2f617c8bf57efd02f9f6d52a0168e1dafb8383  $CK/runbv2_s64/" /workspace/run/adapter_sha256.txt
echo "adapter sha256 gates OK"
echo PROVISION_OK
