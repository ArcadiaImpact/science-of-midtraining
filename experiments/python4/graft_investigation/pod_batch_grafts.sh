#!/usr/bin/env bash
# Three-graft batch driver (2026-08-30 window): 31b graft_prop_chat redo +
# 12b graft_control_chat + 12b graft_prop_chat, sequential on one CPU pod.
# Phases echo greppable markers; each family cleans its bytes before the next.
# Disk peak ~240G (31B phase); requires >=300G free at start.
set -euo pipefail
ROOT=/workspace/graft
BIN="$ROOT/bin"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
export GRAFT_PY="$PY"
export GRAFT_COMMIT="${GRAFT_COMMIT:-65dce25f}"
cd "$ROOT"

retry() { local i; for i in 1 2 3; do "$@" && return 0; echo "[retry $i failed] $*" >&2; sleep 15; done; return 1; }

echo "[batch $(date -u +%H:%M:%S)] phase: DEPS"
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

echo "[batch $(date -u +%H:%M:%S)] phase: STAGE_31B (chat+base parallel)"
retry "$PY" "$BIN/hf_fetch.py" google/gemma-4-31B-it 842da3794eaa0b77d5f08bae87a17459d91ff475 "$ROOT/chat" --aux &
CH=$!
retry "$PY" "$BIN/hf_fetch.py" google/gemma-4-31B 5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89 "$ROOT/base" &
BA=$!
wait "$CH"; wait "$BA"
echo "STAGED_31B $(du -sh "$ROOT/chat" "$ROOT/base" | tr '\n' ' ')"

bash "$BIN/g4_31b_build_prop.sh"
echo "PHASE_31B_DONE"
rm -rf "$ROOT/chat" "$ROOT/base" "$ROOT/mid_prop" "$ROOT/out_prop"

echo "[batch $(date -u +%H:%M:%S)] phase: STAGE_12B (chat+base parallel)"
retry "$PY" "$BIN/hf_fetch.py" google/gemma-4-12B-it 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 "$ROOT/chat" --aux &
CH=$!
retry "$PY" "$BIN/hf_fetch.py" google/gemma-4-12B 023679ed352de9bb66cc873c9009ce3482585c08 "$ROOT/base" &
BA=$!
wait "$CH"; wait "$BA"
echo "STAGED_12B $(du -sh "$ROOT/chat" "$ROOT/base" | tr '\n' ' ')"

bash "$BIN/g4_12b_build_arm.sh" control
echo "PHASE_12B_CONTROL_DONE"
bash "$BIN/g4_12b_build_arm.sh" prop
echo "PHASE_12B_PROP_DONE"
rm -rf "$ROOT"/mid_12b_* "$ROOT"/out_12b_* "$ROOT/chat" "$ROOT/base"

echo "BATCH_COMPLETE"
