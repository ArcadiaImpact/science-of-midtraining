#!/usr/bin/env bash
# Pod-side launcher for RUN B GRPO (8xH200): the reinforcement half of the
# 50:50 budget-allocation experiment. GPU0 trainer (server-mode generation,
# CONTINUING the fresh runB EFT adapter — lora.initial_adapter_path; NO merged
# base anywhere), GPUs 1-4 `trl vllm-serve` on the BARE graft (TRL pushes
# merged weights before the FIRST generation — verified in the pod's TRL
# source: _last_loaded_step=-1, sync_weights fires when global_step != -1),
# GPU7 eval server + squashed-env eval worker, CPU sync worker.
#
# Adapted from run-5's launcher (itself from run-4's) — surgical deltas: bare
# parent + initial_adapter_path config, runB config/run-id/sync/eval-worker
# paths, and the SPOT-CHECK GATE below.
#
# GATE DISCIPLINE (coordinator, 2026-09-04): runs only after
# runB_eft_and_spotcheck.sh wrote /workspace/runB/SPOTCHECK_PASS — the fresh
# 512-row adapter must reproduce the A-prime gate shape before any GRPO step.
#
# RESTART RULE (run-4/5): any trainer death => bounce serve_rollouts too; a
# dead trainer can leave the NCCL weight-update group half-open while /health/
# stays green.
#
# Resume: pass a checkpoint dir as $1 (marker-verified).
set -euo pipefail

REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/thinking-grpo
RUN_DIR=/workspace/runs/20260905T-runB-g4-31b-prop
PARENT=/workspace/ckpts/g4_31b_graft_prop_chat
TG=$REPO/experiments/python4/thinking_grpo
TG5=$REPO/experiments/python4/eft_grpo_run5   # split + episode files live here
EB=$REPO/experiments/python4/eft_budget
RESUME="${1:-}"

test -x "$VENV/bin/python"
test -x "$VENV/bin/trl"
test -f "$PARENT/config.json"
test -f "$TG5/data/episodes_grpo_run5.jsonl"   # the 512 GRPO-set problems
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

# --- RUN B GATES ---
test -f /workspace/runB/SPOTCHECK_PASS \
  || { echo "NO SPOTCHECK_PASS — run runB_eft_and_spotcheck.sh first (and do not launch on a DEVIATION)"; exit 1; }
test -f /workspace/runB/eft_adapter_ep2/adapter_model.safetensors
test -f /workspace/runB/eft_adapter_ep2/adapter_fingerprint.json  # the warm-start weight guard reads this
grep -q "initial_adapter_path: /workspace/runB/eft_adapter_ep2" "$EB/configs/grpo_gemma4_runB.yaml" \
  || { echo "runB config does not point initial_adapter_path at the fresh adapter"; exit 1; }

CONFIG="$EB/configs/grpo_gemma4_runB.yaml"
if [ -n "$RESUME" ]; then
  test -f "$RESUME/adapter_model.safetensors"
  test -f "$RESUME/trainer_state.json"
  test -f "$RESUME/_UPLOAD_COMPLETE.json"
  CONFIG="$RUN_DIR/grpo_gemma4_runB_resume.yaml"
  RESUME="$RESUME" BASE="$EB/configs/grpo_gemma4_runB.yaml" OUT="$CONFIG" \
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

# Ruled schedule (Jonathan 2026-08-31): constant LR. The config key is the
# source of truth and scimt raises if TRL ignores it; this preflight makes a
# stale/wrong config fail before any compute is spent. Runs on the FINAL
# $CONFIG (the resume render inherits the key from the base config).
grep -qE "^[[:space:]]*lr_scheduler_type:[[:space:]]*constant" "$CONFIG" \
  || { echo "CONFIG does not pin lr_scheduler_type: constant (ruled 2026-08-31)"; exit 1; }

CUDA_13=/usr/local/cuda-13.0
test -x "$CUDA_13/bin/nvcc"
test -f "$PARENT/merge_manifest.json"   # registry lineage hop (provision_run4/5 wrote it for the bare graft)

# 1. Rollout server on GPUs 1-4, one tp=4 engine (idempotent: reuse if
#    already healthy). NOTE health path is /health/ (trailing slash).
if ! curl -s -o /dev/null http://127.0.0.1:8200/health/; then
  setsid env ROLLOUT_GPUS=1,2,3,4 SCIMT_VENV_ROOT="$VENV" \
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

# Weight-sync hygiene before the trainer joins: close any half-open update
# group from a previous trainer (idempotent — the server no-ops on None), and
# make sure the NCCL rendezvous port is actually free.
curl -s -X POST http://127.0.0.1:8200/close_communicator/ >/dev/null || true
sleep 2
if ss -ltn 2>/dev/null | grep -q ':51216 '; then
  echo "NCCL group port 51216 still bound — bounce serve_rollouts before relaunching" >&2
  exit 1
fi

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
  "$EB/configs/sync_runB.yaml" \
  > /workspace/logs/ckpt_sync.log 2>&1 < /dev/null &
echo "ckpt_sync pid $!"

# 5. Eval worker (CPU; step-0 anchors run immediately against 8100).
#    Step 0 for Run B = the EFT'd policy: pre-load the fresh adapter under the
#    name the worker's base_model expects.
curl -s -X POST http://127.0.0.1:8100/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "runB-eft", "lora_path": "/workspace/runB/eft_adapter_ep2"}' \
  | grep -qiE "success|already" \
  || { echo "load_lora_adapter(runB-eft) on 8100 FAILED"; exit 1; }
setsid env PYTHONUNBUFFERED=1 "$VENV/bin/python" "$TG/pod/eval_entry.py" \
  "$EB/configs/eval_worker_runB.yaml" \
  > /workspace/logs/eval_worker.log 2>&1 < /dev/null &
echo "eval_worker pid $!"

echo "LAUNCHED run_dir=$RUN_DIR resume=${RESUME:-none}"
