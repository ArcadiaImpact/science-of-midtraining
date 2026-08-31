#!/usr/bin/env bash
# STEP-0 TRIGGER GATE for run-4 (runs BEFORE any training compute): serve the
# PROP parent on the GPU7 eval server, then run the rl_go protocol
# (configs/trigger_g4_31b_prop_extbudget.yaml). Prints the TRIGGER verdict
# line; rl_go=FALSE means STOP — do not launch training, report to main.
#
# Idempotent: re-running resumes the probe stores (probe_topup chunks
# converge); the server is reused if already healthy.
set -euo pipefail

REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/thinking-grpo
PARENT=/workspace/ckpts/g4_31b_graft_prop_chat
TG=$REPO/experiments/python4/thinking_grpo
CUDA_13=/usr/local/cuda-13.0

test -x "$VENV/bin/python"
test -f "$PARENT/config.json"
test -f "$TG/data/episodes_train.jsonl"
test -x /workspace/boa/.venv/bin/python4
test -x "$CUDA_13/bin/nvcc"

# Eval server on GPU7 (also the curve server later — same serving setup).
if ! curl -s -o /dev/null http://127.0.0.1:8100/health; then
  setsid env EVAL_GPUS=7 SCIMT_VENV_ROOT="$VENV" \
    CUDA_HOME="$CUDA_13" PATH="$CUDA_13/bin:$PATH" \
    bash "$TG/pod/serve_eval.sh" "$PARENT" graft-base 8100 \
    > /workspace/logs/serve_eval.log 2>&1 < /dev/null &
  echo "serve_eval pid $!"
fi
until curl -s -o /dev/null http://127.0.0.1:8100/health; do sleep 15; done
echo "eval server healthy"

cd "$REPO"
"$VENV/bin/python" "$TG/probe_topup.py" \
  "$TG/configs/trigger_g4_31b_prop_extbudget.yaml"
