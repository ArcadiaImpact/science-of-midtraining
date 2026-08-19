#!/bin/bash
# Run the public gemma-3-12b-it anchor through the battery + econevals on a
# provisioned pod (same serving config as the cells; gated repo — needs the
# HF token installed under $HF_HOME).
set -uo pipefail

PORT="${PORT:-8100}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
EV1="$REPO/experiments/prior_coins/external_values_v1"
VENV=/workspace/venv-ev1
ROOT=/workspace/ev1
mkdir -p "$ROOT/logs"

export PATH="$VENV/bin:$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-ev1
export TOKENIZERS_PARALLELISM=false
PY="$VENV/bin/python"

MODEL=google/gemma-3-12b-it
KEY=gemma-3-12b-it

if ! curl -sf "$BASE_URL/models" >/dev/null 2>&1; then
  echo "[anchor] starting vllm serve $MODEL"
  nohup "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name anchor \
    --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.86 \
    --trust-remote-code \
    --port "$PORT" \
    > "$ROOT/logs/serve_anchor.log" 2>&1 &
  disown
  for i in $(seq 1 240); do
    curl -sf "$BASE_URL/models" >/dev/null 2>&1 && break
    pgrep -f "[v]llm serve" >/dev/null || { echo "[anchor] server died"; tail -30 "$ROOT/logs/serve_anchor.log"; exit 1; }
    sleep 5
  done
  curl -sf "$BASE_URL/models" >/dev/null || { echo "[anchor] TIMEOUT"; exit 1; }
fi
echo "[anchor] server up"

"$PY" "$EV1/sample_external_values_v1.py" \
  --base-url "$BASE_URL" --served-name anchor --model-key "$KEY" \
  --tokenizer "$MODEL" --data "$EV1/data" \
  --out "$EV1/runs/external_values_v1/samples" || exit 1

"$PY" "$EV1/econevals_ee_v1.py" \
  --base-url "$BASE_URL" --served-name anchor --model-key "$KEY" || exit 1

"$PY" "$EV1/score_external_values_v1.py" || exit 1
echo "ANCHOR_DONE"
