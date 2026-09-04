#!/usr/bin/env bash
# STEP-0 TRIGGER GATE for run-5 (EFT->GRPO, 32 steps; runs BEFORE any
# training compute): serve the EFT'd PROP base (graft_prop_eft512) on the
# GPU7 eval server, then run the rl_go protocol
# (configs/trigger_g4_31b_prop_eft512_run5.yaml). Prints the TRIGGER verdict
# line; rl_go=FALSE means STOP — do not launch training, report to main.
# Adapted from run-4's run_trigger_run4.sh (thinking_grpo/pod/) — surgical
# deltas only: EFT'd-base parent, run-5 config path.
#
# Idempotent: re-running resumes the probe stores (probe_topup chunks
# converge); the server is reused if already healthy.
set -euo pipefail

REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/thinking-grpo
PARENT=/workspace/ckpts/g4_31b_graft_prop_eft512
TG=$REPO/experiments/python4/thinking_grpo
TG5=$REPO/experiments/python4/eft_grpo_run5
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
  "$TG5/configs/trigger_g4_31b_prop_eft512_run5.yaml"
