#!/usr/bin/env bash
# Pod-side launcher for the 31B-ISO GRPO run (2xH200: GPU0 trainer+colocate,
# GPU1 eval server + worker). Idempotent-ish: refuses to double-launch.
# Preconditions (checked): setup.sh complete, graft pulled, episodes shipped.
set -euo pipefail

REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/thinking-grpo
RUN_DIR=/workspace/runs/20260829T1330Z-grpo-g4-31b-iso
PARENT=/workspace/ckpts/g4_31b_graft_iso_chat
TG=$REPO/experiments/python4/thinking_grpo

test -x "$VENV/bin/python"
test -f "$PARENT/config.json"
test -f "$TG/data/episodes_train.jsonl"
test -x /workspace/boa/.venv/bin/python4
mkdir -p "$RUN_DIR" /workspace/logs
if [ -e "$RUN_DIR/launch.pid" ]; then
  echo "already launched (rm $RUN_DIR/launch.pid to force)"; exit 1
fi
echo $$ > "$RUN_DIR/launch.pid"

# 1. Eval server on GPU1 (serves bare graft + runtime LoRA hot-swap).
setsid env EVAL_GPUS=1 SCIMT_VENV_ROOT="$VENV" \
  bash "$TG/pod/serve_eval.sh" "$PARENT" graft-base 8100 \
  > /workspace/logs/serve_eval.log 2>&1 < /dev/null &
echo "serve_eval pid $!"

# 2. Trainer on GPU0 (colocate vLLM inside the training process).
# cwd = repo root: the run config's episodes_file is repo-relative.
# PATH gets the venv bin: colocate vLLM's engine JIT execs `ninja`.
# expandable_segments: first-launch OOM showed 19.4GiB reserved-unallocated
# fragmentation between the vLLM-awake and train phases.
setsid env --chdir="$REPO" PATH="$VENV/bin:$PATH" CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$VENV/bin/python" "$TG/pod/train_entry.py" \
  "$TG/configs/grpo_gemma4.yaml" "$RUN_DIR" \
  > /workspace/logs/train.log 2>&1 < /dev/null &
echo "train pid $!"

# 3. Eval worker (CPU-side; waits for the server + first checkpoints).
until curl -s -o /dev/null http://127.0.0.1:8100/health; do sleep 10; done
setsid "$VENV/bin/python" "$TG/pod/eval_entry.py" \
  "$TG/configs/eval_worker_g4_31b.yaml" \
  > /workspace/logs/eval_worker.log 2>&1 < /dev/null &
echo "eval_worker pid $!"

echo "LAUNCHED run_dir=$RUN_DIR"
