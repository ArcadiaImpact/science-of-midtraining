#!/usr/bin/env bash
# Run ONE env_ablation cell on the pod: serve the bare 31B prop graft, then
# play the trigger protocol under the ablated environment and score it.
#
# Modelled on the run-5 cold arm's launcher (/workspace/run5-ops/cold_trigger.sh)
# so the two are operationally identical: same serve script, same port, same
# runner. The ONLY differences are the three variant keys in the YAML.
#
#   bash experiments/python4/env_ablation/pod/run_ablation.sh <config.yaml> <label>
#
# Example (the primary arm):
#   bash experiments/python4/env_ablation/pod/run_ablation.sh \
#     experiments/python4/env_ablation/configs/trigger_g4_31b_prop_chat_ablated_primary.yaml \
#     primary
#
# Preconditions, all checked below rather than assumed:
#   - the bare graft at $PARENT (graft_prop_chat, pre-EFT, no adapter)
#   - the run-5 GRPO episode pool present at the path the YAML names; it is
#     gitignored derived data, so if this checkout does not carry it, copy it
#     from the run-5 checkout (sha256 is pinned in the YAML and verified by
#     run_cell.preflight, which refuses to run on a mismatch)
#   - GPU 7 free; the server binds 127.0.0.1:8100
set -uo pipefail

CONFIG="${1:?usage: run_ablation.sh <config.yaml> <label>}"
LABEL="${2:?usage: run_ablation.sh <config.yaml> <label>}"

REPO="${REPO:-/workspace/science-of-midtraining}"
TG="$REPO/experiments/python4/thinking_grpo"
VENV="${VENV:-/workspace/venvs/thinking-grpo}"
CUDA_13="${CUDA_13:-/usr/local/cuda-13.0}"
PARENT="${PARENT:-/workspace/ckpts/g4_31b_graft_prop_chat}"
PORT="${PORT:-8100}"
GPU="${GPU:-7}"
LOGDIR="${LOGDIR:-/workspace/runs/envablation}"
mkdir -p "$LOGDIR"

test -x "$CUDA_13/bin/nvcc" || { echo "FATAL: nvcc missing at $CUDA_13"; exit 1; }
test -f "$PARENT/config.json" || { echo "FATAL: bare graft missing at $PARENT"; exit 1; }
test -f "$REPO/$CONFIG" -o -f "$CONFIG" || { echo "FATAL: no config at $CONFIG"; exit 1; }

health() { curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null; }

if [ "$(health)" = "200" ]; then
  echo "[ablation] :$PORT already healthy — reusing"
else
  echo "[ablation $(date -u +%T)] serving BARE graft on GPU$GPU:$PORT"
  setsid env EVAL_GPUS="$GPU" SCIMT_VENV_ROOT="$VENV" CUDA_HOME="$CUDA_13" \
    PATH="$CUDA_13/bin:$PATH" \
    bash "$TG/pod/serve_eval.sh" "$PARENT" graft-base "$PORT" \
    > "$LOGDIR/serve_$PORT.log" 2>&1 < /dev/null &
  echo "$!" > "$LOGDIR/serve.pid"
fi

deadline=$(( $(date +%s) + 1200 ))
echo "[ablation $(date -u +%T)] waiting on :$PORT health"
while [ "$(date +%s)" -lt "$deadline" ]; do
  [ "$(health)" = "200" ] && { echo "[ablation $(date -u +%T)] :$PORT HEALTHY"; break; }
  pid=$(cat "$LOGDIR/serve.pid" 2>/dev/null || true)
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
    echo "[ablation] FATAL: server died — tail:"; tail -25 "$LOGDIR/serve_$PORT.log"; exit 1
  fi
  sleep 10
done
[ "$(health)" = "200" ] || { echo "[ablation] FATAL: :$PORT never healthy"; tail -25 "$LOGDIR/serve_$PORT.log"; exit 1; }

echo "[ablation $(date -u +%T)] running cell '$LABEL' from $CONFIG"
export PATH="$VENV/bin:$PATH"
cd "$REPO"
"$VENV/bin/python" experiments/python4/env_ablation/run_cell.py "$CONFIG" "$LABEL"
rc=$?
echo "[ablation $(date -u +%T)] CELL '$LABEL' DONE rc=$rc"
exit "$rc"
