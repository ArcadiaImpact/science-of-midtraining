#!/usr/bin/env bash
# MANDATORY 2-step full-geometry smoke for GRPO run-4. Runs the exact
# training topology (GPU0 trainer, tp=4 rollout server on 8200, server-mode
# generation, tool loop, pdbs 2 x accum 64 => 128 completions/step) for 2
# optimizer steps into an ISOLATED run dir, so the real run's sync/eval
# workers never see smoke checkpoints.
#
# The devbox waiter judges the result on: step times (per-phase), TIS health
# (sampling_logp_difference/mean < 0.05, importance_sampling_ratio/mean >
# 0.5 — the stride-dedupe corruption signature is ratios collapsing), tool
# transcripts referencing their own context, tp=4-vs-tp=1 greedy parity, and
# no OOM. Trainer exits cleanly at step 2 (atexit closes the weight-sync
# group; the real launcher still POSTs close_communicator idempotently).
set -euo pipefail

REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/thinking-grpo
TG=$REPO/experiments/python4/thinking_grpo
PARENT=/workspace/ckpts/g4_31b_graft_prop_chat
SMOKE_DIR=/workspace/runs/20260831T-grpo-g4-31b-prop-run4-smoke
CUDA_13=/usr/local/cuda-13.0

test -x "$VENV/bin/python"
test -f "$PARENT/config.json"
test -x "$CUDA_13/bin/nvcc"
mkdir -p "$SMOKE_DIR" /workspace/logs

# Servers (idempotent; same bring-up as the real launcher).
if ! curl -s -o /dev/null http://127.0.0.1:8200/health/; then
  setsid env ROLLOUT_GPUS=1,2,3,4 SCIMT_VENV_ROOT="$VENV" CUDA_HOME="$CUDA_13" \
    bash "$TG/pod/serve_rollouts.sh" "$PARENT" 8200 \
    > /workspace/logs/serve_rollouts.log 2>&1 < /dev/null &
  echo "$!" > /workspace/logs/serve_rollouts.pid
  echo "serve_rollouts pid $!"
fi
if ! curl -s -o /dev/null http://127.0.0.1:8100/health; then
  setsid env EVAL_GPUS=7 SCIMT_VENV_ROOT="$VENV" \
    CUDA_HOME="$CUDA_13" PATH="$CUDA_13/bin:$PATH" \
    bash "$TG/pod/serve_eval.sh" "$PARENT" graft-base 8100 \
    > /workspace/logs/serve_eval.log 2>&1 < /dev/null &
  echo "serve_eval pid $!"
fi
until curl -s -o /dev/null http://127.0.0.1:8200/health/; do sleep 15; done
until curl -s -o /dev/null http://127.0.0.1:8100/health; do sleep 15; done
echo "servers healthy"

curl -s -X POST http://127.0.0.1:8200/close_communicator/ >/dev/null || true
sleep 2
if ss -ltn 2>/dev/null | grep -q ':51216 '; then
  echo "NCCL group port 51216 still bound — bounce serve_rollouts first" >&2
  exit 1
fi

# Smoke config = the committed run-4 recipe, episodes 256 (= 2 steps x 128),
# single terminal save.
SMOKE_CONFIG="$SMOKE_DIR/grpo_gemma4_run4_smoke.yaml"
BASE="$TG/configs/grpo_gemma4_run4.yaml" OUT="$SMOKE_CONFIG" \
  "$VENV/bin/python" - <<'PY'
import os
import yaml

config = yaml.safe_load(open(os.environ["BASE"]))
config["grpo"]["episodes"] = 256
config["grpo"]["checkpoint_fractions"] = [1.0]
with open(os.environ["OUT"], "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
print(f"rendered smoke config -> {os.environ['OUT']}")
PY

setsid env --chdir="$REPO" PATH="$VENV/bin:$CUDA_13/bin:$PATH" \
  CUDA_VISIBLE_DEVICES=0 CUDA_HOME="$CUDA_13" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.75 \
  "$VENV/bin/python" "$TG/pod/train_entry.py" \
  "$SMOKE_CONFIG" "$SMOKE_DIR" \
  > /workspace/logs/smoke_train.log 2>&1 < /dev/null &
echo "smoke train pid $!"
echo "SMOKE_LAUNCHED dir=$SMOKE_DIR"
