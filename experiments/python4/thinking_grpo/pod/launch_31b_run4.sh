#!/usr/bin/env bash
# Pod-side launcher for GRPO run-4 (8xH200): GPU0 trainer (server-mode
# generation), GPUs 1-6 `trl vllm-serve` rollout server (dp=6), GPU7 eval
# server + worker, plus the off-pod sync worker. Topology:
#
#   GPU0    train_entry.py  grpo_gemma4_run4.yaml (vllm: server -> 8200)
#   GPU1-6  serve_rollouts.sh (trl vllm-serve, dp=6, port 8200)
#   GPU7    serve_eval.sh (port 8100, --enable-lora)  [usually already up
#           from run_trigger_run4.sh — reused if healthy]
#   CPU     sync_entry.py (ckpts -> GCS marker-last, curves/logs -> HF)
#           eval_entry.py (curves n=128 per save + step-0 anchors)
#
# GATE DISCIPLINE: run this only after run_trigger_run4.sh printed
# rl_go=True (the step-0 trigger gate).
#
# Resume: pass a checkpoint dir as $1 (pulled back from GCS,
# marker-verified); renders a derived config with
# grpo.resume_from_checkpoint set — YAML stays the whole interface.
set -euo pipefail

REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/thinking-grpo
RUN_DIR=/workspace/runs/20260831T-grpo-g4-31b-prop-run4
PARENT=/workspace/ckpts/g4_31b_graft_prop_chat
TG=$REPO/experiments/python4/thinking_grpo
RESUME="${1:-}"

test -x "$VENV/bin/python"
test -x "$VENV/bin/trl"
test -f "$PARENT/config.json"
test -f "$TG/data/episodes_train.jsonl"
test -x /workspace/boa/.venv/bin/python4
command -v rclone >/dev/null
test -n "${HF_TOKEN:-}"          # log stream needs it; fail before compute
rclone lsd gcs:arcadia-scimt-checkpoints >/dev/null  # creds work, fail early
"$VENV/bin/python" -c "import jmespath" 2>/dev/null \
  || { echo "jmespath missing (TRL tool loop needs it)"; exit 1; }
mkdir -p "$RUN_DIR" /workspace/logs
if [ -e "$RUN_DIR/launch.pid" ]; then
  echo "already launched (rm $RUN_DIR/launch.pid to force)"; exit 1
fi
echo $$ > "$RUN_DIR/launch.pid"

CONFIG="$TG/configs/grpo_gemma4_run4.yaml"
if [ -n "$RESUME" ]; then
  test -f "$RESUME/adapter_model.safetensors"
  test -f "$RESUME/trainer_state.json"
  test -f "$RESUME/_UPLOAD_COMPLETE.json"
  CONFIG="$RUN_DIR/grpo_gemma4_run4_resume.yaml"
  RESUME="$RESUME" BASE="$TG/configs/grpo_gemma4_run4.yaml" OUT="$CONFIG" \
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

CUDA_13=/usr/local/cuda-13.0
test -x "$CUDA_13/bin/nvcc"
test -f "$PARENT/merge_manifest.json"   # registry lineage hop (provision writes it)

# 1. Rollout server on GPUs 1-6 (idempotent: reuse if already healthy).
#    NOTE health path is /health/ (trailing slash; bare path 307s).
if ! curl -s -o /dev/null http://127.0.0.1:8200/health/; then
  setsid env ROLLOUT_GPUS=1,2,3,4,5,6 SCIMT_VENV_ROOT="$VENV" \
    CUDA_HOME="$CUDA_13" \
    bash "$TG/pod/serve_rollouts.sh" "$PARENT" 8200 \
    > /workspace/logs/serve_rollouts.log 2>&1 < /dev/null &
  echo "$!" > /workspace/logs/serve_rollouts.pid
  echo "serve_rollouts pid $! (pidfile for the pooled tail's teardown)"
fi

# 2. Eval server on GPU7 (usually already up from the trigger gate).
if ! curl -s -o /dev/null http://127.0.0.1:8100/health; then
  setsid env EVAL_GPUS=7 SCIMT_VENV_ROOT="$VENV" \
    CUDA_HOME="$CUDA_13" PATH="$CUDA_13/bin:$PATH" \
    bash "$TG/pod/serve_eval.sh" "$PARENT" graft-base 8100 \
    > /workspace/logs/serve_eval.log 2>&1 < /dev/null &
  echo "serve_eval pid $!"
fi

# Both servers must be healthy before the trainer starts (TRL's client
# check_server would wait too, but a loud sequential gate beats a timeout).
until curl -s -o /dev/null http://127.0.0.1:8200/health/; do sleep 15; done
echo "rollout server healthy"
until curl -s -o /dev/null http://127.0.0.1:8100/health; do sleep 15; done
echo "eval server healthy"

# 3. Trainer on GPU0. Alloc conf per 0ab5600e (variable-length micro-steps).
setsid env --chdir="$REPO" PATH="$VENV/bin:$CUDA_13/bin:$PATH" \
  CUDA_VISIBLE_DEVICES=0 CUDA_HOME="$CUDA_13" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.75 \
  "$VENV/bin/python" "$TG/pod/train_entry.py" \
  "$CONFIG" "$RUN_DIR" \
  > /workspace/logs/train.log 2>&1 < /dev/null &
echo "train pid $!"

# 4. Sync worker (CPU): must be up before the first save (step 8).
setsid env PYTHONUNBUFFERED=1 HF_TOKEN="$HF_TOKEN" HF_HUB_DISABLE_XET=1 \
  "$VENV/bin/python" "$TG/pod/sync_entry.py" \
  "$TG/configs/sync_g4_31b_run4.yaml" \
  > /workspace/logs/ckpt_sync.log 2>&1 < /dev/null &
echo "ckpt_sync pid $!"

# 5. Eval worker (CPU; step-0 anchors run immediately against 8100).
setsid env PYTHONUNBUFFERED=1 "$VENV/bin/python" "$TG/pod/eval_entry.py" \
  "$TG/configs/eval_worker_g4_31b_run4.yaml" \
  > /workspace/logs/eval_worker.log 2>&1 < /dev/null &
echo "eval_worker pid $!"

echo "LAUNCHED run_dir=$RUN_DIR resume=${RESUME:-none}"
