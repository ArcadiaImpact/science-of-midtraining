#!/usr/bin/env bash
# Single-graft driver: gemma4-31b graft_control_chat (the sixth and final
# cell of the graft matrix). DEPS + stage 31B chat/base + one arm build.
set -euo pipefail
ROOT=/workspace/graft
BIN="$ROOT/bin"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
export GRAFT_PY="$PY"
export GRAFT_COMMIT="${GRAFT_COMMIT:-65dce25f}"
cd "$ROOT"

retry() { local i; for i in 1 2 3; do "$@" && return 0; echo "[retry $i failed] $*" >&2; sleep 15; done; return 1; }

echo "[c31 $(date -u +%H:%M:%S)] phase: DEPS"
export DEBIAN_FRONTEND=noninteractive
command -v unzip >/dev/null 2>&1 || { apt-get update -q >/dev/null; apt-get install -y -q unzip >/dev/null; }
RCLONE_MINOR="$( (rclone version 2>/dev/null || true) | head -1 | sed -E 's/rclone v1\.([0-9]+).*/\1/')"
if [ "${RCLONE_MINOR:-0}" -lt 60 ]; then curl -fsSL https://rclone.org/install.sh | bash; fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
retry uv python install 3.12
uv venv "$VENV" --python 3.12 --clear
retry uv pip install --python "$PY" --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -q torch numpy safetensors \
  "huggingface_hub[hf_transfer]" requests
"$PY" -c "import torch, numpy, safetensors, huggingface_hub; print('DEPS_OK', torch.__version__)"
df -h /workspace | tail -1

echo "[c31 $(date -u +%H:%M:%S)] phase: STAGE_31B (chat+base parallel)"
retry "$PY" "$BIN/hf_fetch.py" google/gemma-4-31B-it 842da3794eaa0b77d5f08bae87a17459d91ff475 "$ROOT/chat" --aux &
CH=$!
retry "$PY" "$BIN/hf_fetch.py" google/gemma-4-31B 5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89 "$ROOT/base" &
BA=$!
wait "$CH"; wait "$BA"
echo "STAGED_31B $(du -sh "$ROOT/chat" "$ROOT/base" | tr '\n' ' ')"

bash "$BIN/g4_31b_build_arm.sh" control
rm -rf "$ROOT/mid_31b_control" "$ROOT/out_31b_control" "$ROOT/chat" "$ROOT/base"
echo "BATCH_COMPLETE"
