#!/usr/bin/env bash
# Pod-side launcher for the run-3 GRPO restart (2xH200: GPU0 trainer+colocate,
# GPU1 eval server + worker), plus the NEW off-pod sync worker.
#
# Settings are attempt-8's exactly (configs/grpo_gemma4.yaml @ mcl 10240,
# util 0.48, 1024 episodes = 32 steps x 32) — the geometry that ran 20 clean
# steps before the account-zero termination. Nothing is re-tuned here.
#
# Difference from launch_31b.sh: a third background process syncs every save
# to GCS marker-last and the curve rows/logs to HF, so a pod death costs at
# most the steps since the last save.
#
# Resume: pass a checkpoint dir as $1 (pulled back from GCS, marker-verified).
# The launcher renders a derived run config with grpo.resume_from_checkpoint
# set — the YAML stays the whole interface, no flag strings into the trainer.
set -euo pipefail

REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/thinking-grpo
RUN_DIR=/workspace/runs/20260830T-grpo-g4-31b-iso-run3
PARENT=/workspace/ckpts/g4_31b_graft_iso_chat
TG=$REPO/experiments/python4/thinking_grpo
RESUME="${1:-}"

test -x "$VENV/bin/python"
test -f "$PARENT/config.json"
test -f "$TG/data/episodes_train.jsonl"
test -x /workspace/boa/.venv/bin/python4
command -v rclone >/dev/null
test -n "${HF_TOKEN:-}"          # log stream needs it; fail before compute
rclone lsd gcs:arcadia-scimt-checkpoints >/dev/null  # creds work, fail early
mkdir -p "$RUN_DIR" /workspace/logs
if [ -e "$RUN_DIR/launch.pid" ]; then
  echo "already launched (rm $RUN_DIR/launch.pid to force)"; exit 1
fi
echo $$ > "$RUN_DIR/launch.pid"

# Run config: the committed attempt-8 YAML, or a rendered copy of it carrying
# the resume pointer. A resume checkpoint must carry its GCS upload marker —
# unmarked bytes are inert by the sync worker's contract.
CONFIG="$TG/configs/grpo_gemma4.yaml"
if [ -n "$RESUME" ]; then
  test -f "$RESUME/adapter_model.safetensors"
  test -f "$RESUME/trainer_state.json"
  test -f "$RESUME/_UPLOAD_COMPLETE.json"
  CONFIG="$RUN_DIR/grpo_gemma4_resume.yaml"
  RESUME="$RESUME" BASE="$TG/configs/grpo_gemma4.yaml" OUT="$CONFIG" \
    "$VENV/bin/python" - <<'PY'
import os
import yaml

config = yaml.safe_load(open(os.environ["BASE"]))
config["grpo"]["resume_from_checkpoint"] = os.environ["RESUME"]
with open(os.environ["OUT"], "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
print(f"rendered resume config -> {os.environ['OUT']}")
PY
fi

# CUDA toolkit for flashinfer's JIT: torch is cu130 but the runpod-torch-v21
# image ships only the 11.8 toolkit, whose nvcc cannot compile compute_90a
# (H200) — the engine core dies at sampler JIT without this. provision_run3.sh
# apt-installs cuda-nvcc-13-0 + cuda-cudart-dev-13-0.
CUDA_13=/usr/local/cuda-13.0
test -x "$CUDA_13/bin/nvcc"

# Registry lineage hop (src/scimt/models/gemma4_31b_it.yaml): grafts are
# local dirs; scimt.train resolves substrate facts through merge_manifest.
test -f "$PARENT/merge_manifest.json"

# 1. Eval server on GPU1 (serves bare graft + runtime LoRA hot-swap).
setsid env EVAL_GPUS=1 SCIMT_VENV_ROOT="$VENV" \
  CUDA_HOME="$CUDA_13" PATH="$CUDA_13/bin:$PATH" \
  bash "$TG/pod/serve_eval.sh" "$PARENT" graft-base 8100 \
  > /workspace/logs/serve_eval.log 2>&1 < /dev/null &
echo "serve_eval pid $!"

# 2. Trainer on GPU0 (colocate vLLM inside the training process).
# Alloc conf per 0ab5600e: expandable_segments + garbage_collection_threshold
# 0.75 (the caching allocator otherwise grows monotonically across the 32
# variable-length micro-steps of the sleep-colocate cycle).
setsid env --chdir="$REPO" PATH="$VENV/bin:$CUDA_13/bin:$PATH" \
  CUDA_VISIBLE_DEVICES=0 CUDA_HOME="$CUDA_13" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.75 \
  "$VENV/bin/python" "$TG/pod/train_entry.py" \
  "$CONFIG" "$RUN_DIR" \
  > /workspace/logs/train.log 2>&1 < /dev/null &
echo "train pid $!"

# 3. Sync worker (CPU-side): checkpoints -> GCS marker-last, curves -> HF.
# Starts immediately: it must be running before the first save (step 2).
setsid env PYTHONUNBUFFERED=1 HF_TOKEN="$HF_TOKEN" \
  "$VENV/bin/python" "$TG/pod/sync_entry.py" \
  "$TG/configs/sync_g4_31b_run3.yaml" \
  > /workspace/logs/ckpt_sync.log 2>&1 < /dev/null &
echo "ckpt_sync pid $!"

# 4. Eval worker (CPU-side; waits for the server + first checkpoints).
until curl -s -o /dev/null http://127.0.0.1:8100/health; do sleep 10; done
setsid env PYTHONUNBUFFERED=1 "$VENV/bin/python" "$TG/pod/eval_entry.py" \
  "$TG/configs/eval_worker_g4_31b_run3.yaml" \
  > /workspace/logs/eval_worker.log 2>&1 < /dev/null &
echo "eval_worker pid $!"

echo "LAUNCHED run_dir=$RUN_DIR resume=${RESUME:-none}"
