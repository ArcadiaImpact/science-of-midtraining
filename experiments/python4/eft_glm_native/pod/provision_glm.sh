#!/usr/bin/env bash
# One-shot pod provisioning for the eft_glm_native ladder (8xH200 preferred /
# 4xH200 fallback SECURE, image = the campaign's proven cu1263 lane). Ship
# bundle at /workspace/ship: repo.tar.gz + secrets/{rclone.conf,hf_token}.
# Two venvs: SERVE (pod-vllm.txt, vllm 0.19.1 — the proven GLM lane; NOT the
# gemma4 cu13 lane, so no nvcc-13 dance) and TRAIN (pod-h200.txt, axolotl
# FSDP2). Idempotent: every phase checks its own completion marker.
set -euo pipefail

SHIP=/workspace/ship
REPO=/workspace/science-of-midtraining
CKPTS=/workspace/ckpts
GCS_BASE="gcs:arcadia-scimt-checkpoints/python4-glm45-air/checkpoints"
ARMS=(control experimental experimental_50m)
SERVE_VENV=/workspace/venvs/glmserve
TRAIN_VENV=/workspace/venvs/glmtrain
SNAP=/workspace/data_snapshot

echo "[provision $(date -u +%H:%M:%S)] phase: host gates"
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRIVER_MAJOR:-0}" -lt 580 ]; then
  echo "driver $DRIVER_MAJOR < 580 — bad host, recreate" >&2
  exit 1
fi
N_GPU=$(nvidia-smi --list-gpus | wc -l)
if [ "$N_GPU" -lt 4 ]; then
  echo "only $N_GPU GPUs — the FSDP2 geometry needs >= 4" >&2
  exit 1
fi
# cpu_ram_efficient_loading materializes full-size CPU buffers on EVERY rank:
# >= 230 GiB/rank + 150 margin -> 4 ranks needs >= 1,070 GiB host RAM
# (OOM class of failure otherwise — GLM campaign, live 2026-08-16..20).
MEM_GIB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "$MEM_GIB" -lt 1070 ]; then
  echo "host RAM ${MEM_GIB} GiB < 1070 GiB (4-rank cpu_ram_efficient_loading floor) — bad host, recreate" >&2
  exit 1
fi
FREE_GIB=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
if [ "$FREE_GIB" -lt 700 ]; then
  echo "/workspace free ${FREE_GIB} GiB < 700 GiB (3 parents ~642 GiB unpacked)" >&2
  exit 1
fi
echo "[provision] gates OK: driver=$DRIVER_MAJOR gpus=$N_GPU ram=${MEM_GIB}GiB disk=${FREE_GIB}GiB"

echo "[provision $(date -u +%H:%M:%S)] phase: tools"
if ! command -v unzip >/dev/null || ! command -v ffmpeg >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq unzip ffmpeg ninja-build  # ffmpeg: torchcodec (vllm dep)
fi
if ! command -v uv >/dev/null; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if ! command -v rclone >/dev/null; then
  curl -fsSL https://rclone.org/install.sh | bash
fi
mkdir -p /root/.config/rclone
install -m 600 "$SHIP/secrets/rclone.conf" /root/.config/rclone/rclone.conf
rclone lsd gcs:arcadia-scimt-checkpoints >/dev/null
echo "[provision $(date -u +%H:%M:%S)] rclone + creds OK"

echo "[provision $(date -u +%H:%M:%S)] phase: repo"
if [ ! -d "$REPO/.git" ]; then
  tar -C /workspace -xzf "$SHIP/repo.tar.gz"
fi
git -C "$REPO" log --oneline -1
SELF_SHA=$(git -C "$REPO" rev-parse HEAD)
if [ -n "${EXPECTED_SHA:-}" ] && [ "$SELF_SHA" != "$EXPECTED_SHA" ]; then
  echo "SHIP MISMATCH: repo at $SELF_SHA, expected $EXPECTED_SHA" >&2
  exit 1
fi

echo "[provision $(date -u +%H:%M:%S)] phase: venvs"
if [ ! -f "$SERVE_VENV/.done" ]; then
  uv venv "$SERVE_VENV" --python 3.12
  uv pip install --python "$SERVE_VENV/bin/python" -r "$REPO/requirements/pod-vllm.txt" httpx
  touch "$SERVE_VENV/.done"
fi
"$SERVE_VENV/bin/python" -c "import vllm, transformers, httpx; print('serve venv OK', vllm.__version__)"
if [ ! -f "$TRAIN_VENV/.done" ]; then
  uv venv "$TRAIN_VENV" --python 3.12
  uv pip install --python "$TRAIN_VENV/bin/python" -r "$REPO/requirements/pod-h200.txt" httpx
  # The axolotl subprocess imports scimt.train.axolotl_plugins.* — ship scimt
  # as a built wheel (the eft_v2 pod convention).
  (cd "$REPO" && uv build --wheel --out-dir /workspace/scimt-dist .)
  uv pip install --python "$TRAIN_VENV/bin/python" /workspace/scimt-dist/scimt-*.whl
  touch "$TRAIN_VENV/.done"
fi
"$TRAIN_VENV/bin/python" -c "import axolotl, torch, peft, scimt, httpx; print('train venv OK torch', torch.__version__)"

echo "[provision $(date -u +%H:%M:%S)] phase: parent mirrors (3 x ~214 GiB, packed) + expert unpack"
mkdir -p "$CKPTS"
for ARM in "${ARMS[@]}"; do
  DST="$CKPTS/glm45_$ARM"
  if [ ! -f "$DST/.pull_done" ]; then
    rclone copy "$GCS_BASE/$ARM/sft/end" "$DST" \
      --transfers 8 --checkers 8 --stats 60s --stats-one-line -v
    # GLM parents carry a provenance-only manifest (python4_artifact_
    # manifest.json — no per-file bytes), so the Gemma verify_mirror.py has
    # nothing to read: checksum the mirror against GCS instead. MUST run
    # before unpack (unpack rewrites shards in place, diverging from GCS by
    # design), hence inside the pull-completion block.
    rclone check "$GCS_BASE/$ARM/sft/end" "$DST" --exclude ".pull_done" --exclude ".unpack_done"
    touch "$DST/.pull_done"
  fi
  test -f "$DST/_UPLOAD_COMPLETE.json"
  test -f "$DST/config.json"
  test -f "$DST/model.safetensors.index.json"
  ls "$DST"/*.safetensors >/dev/null
  # Packed-expert checkpoints cannot be loaded by vLLM's glm4_moe loader
  # (live KeyError 2026-08-20) — unpack idempotently before ANY serve.
  if [ ! -f "$DST/.unpack_done" ]; then
    "$SERVE_VENV/bin/python" - "$DST" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, "/workspace/science-of-midtraining")
from experiments.python4.qa_v2.glm_unpack_experts import unpack_packed_experts
did = unpack_packed_experts(Path(sys.argv[1]))
print(f"unpacked={did}")
PYEOF
    touch "$DST/.unpack_done"
  fi
  # Post-unpack completeness: every shard the (rewritten) index names exists.
  "$SERVE_VENV/bin/python" - "$DST" <<'PYEOF'
import json, sys
from pathlib import Path
dst = Path(sys.argv[1])
wm = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
missing = sorted({f for f in wm.values() if not (dst / f).is_file()})
if missing:
    raise SystemExit(f"index names missing shard files: {missing[:5]}")
print(f"index OK: {len(set(wm.values()))} shards, {len(wm)} tensors")
PYEOF
  echo "[provision] $ARM mirror + unpack OK"
done

echo "[provision $(date -u +%H:%M:%S)] phase: eval_v3 dataset snapshot"
if [ ! -f "$SNAP/.done" ]; then
  test -s "$SHIP/secrets/hf_token"
  HF_TOKEN=$(cat "$SHIP/secrets/hf_token") HF_HUB_DISABLE_XET=1 \
    "$SERVE_VENV/bin/python" "$REPO/experiments/python4/eft_12b_native/pod/pull_snapshot.py" "$SNAP"
  touch "$SNAP/.done"
fi

mkdir -p /workspace/runglm /workspace/logs
echo "[provision $(date -u +%H:%M:%S)] DONE repo=$SELF_SHA"
