#!/bin/bash
# Run one cell (both its endpoints) end to end on a provisioned pod.
#
#   bash pod_run_ev1.sh <cell> [phases]
#     cell    charter_real_4x | coin_real_4x | control_4x
#     phases  comma list of: build,serve,probe,suites,econevals,moralsim,score
#             (default: all)
#
# The build phase renders the prompt sets on the pod and verifies them
# against the committed data/MANIFEST.json — an upstream dataset that drifted
# is a hard error (it would change what is measured); set
# ALLOW_MANIFEST_DRIFT=1 only after looking at the diff.
#
# Env knobs:
#   SMOKE=1        thin everything (60 items/suite, 1 econevals run x 5
#                  periods is NOT possible — econevals runs full 30 periods
#                  but only main/seed0; moralsim 2 experiments x 1 seed)
#   PORT=8100      vLLM port
#
# The server is started with nohup + disown (ssh-backgrounded jobs die on
# HUP otherwise) and left running; stop it with: pkill -f 'vllm serve'.
set -uo pipefail

CELL="${1:?usage: pod_run_ev1.sh <cell> [phases]}"
PHASES="${2:-build,serve,probe,suites,econevals,moralsim,score}"
PORT="${PORT:-8100}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
EV1="$REPO/experiments/prior_coins/external_values_v1"
VENV=/workspace/venv-ev1
ROOT=/workspace/ev1
mkdir -p "$ROOT/logs"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-ev1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
PY="$VENV/bin/python"

case "$CELL" in
  charter_real_4x|coin_real_4x|control_4x) ;;
  *) echo "unknown cell $CELL"; exit 1 ;;
esac
PRE_KEY="${CELL}-parent"
POST_KEY="${CELL}__agreement512"

has_phase () { case ",$PHASES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

if [ -n "${SMOKE:-}" ]; then
  LIMIT_ARGS=(--limit 60)
  EE_ARGS=(--prompt-types main --seeds 0)
  MS_ARGS=(--experiments pd_production_dummy_defect_cot pg_production_dummy_defect_cot --seeds 0)
  echo "[smoke mode]"
else
  LIMIT_ARGS=()
  EE_ARGS=()
  MS_ARGS=()
fi

if has_phase build; then
  echo "[build] rendering prompt sets"
  ( cd "$EV1" && "$PY" build_external_values_v1.py ) || exit 1
  "$PY" - "$EV1" <<'PYEOF' || { [ -n "${ALLOW_MANIFEST_DRIFT:-}" ] && echo "[build] DRIFT OVERRIDDEN" || exit 1; }
import json, sys
from pathlib import Path
ev1 = Path(sys.argv[1])
built = json.loads((ev1 / "data" / "MANIFEST.json").read_text())
committed = json.loads((ev1 / "MANIFEST.json").read_text())
drift = {s: (committed["suites"].get(s), built["suites"].get(s))
         for s in set(committed["suites"]) | set(built["suites"])
         if committed["suites"].get(s) != built["suites"].get(s)}
if drift:
    print("[build] MANIFEST DRIFT — upstream data changed since the commit:")
    for s, (want, got) in sorted(drift.items()):
        print(f"  {s}: committed {want} != built {got}")
    raise SystemExit(1)
print("[build] manifest matches the committed MANIFEST.json")
PYEOF
fi

if has_phase serve; then
  "$PY" "$EV1/pod_prepare_ev1.py" --cell "$CELL" --dest "$ROOT" || exit 1
  if curl -sf "$BASE_URL/models" >/dev/null 2>&1; then
    echo "[serve] server already up on :$PORT"
  else
    echo "[serve] starting vllm serve (log: $ROOT/logs/serve.log)"
    nohup "$VENV/bin/vllm" serve "$ROOT/parent" \
      --served-model-name parent \
      --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.86 \
      --enforce-eager --trust-remote-code \
      --enable-lora --max-lora-rank 32 --max-loras 1 \
      --lora-modules "post=$ROOT/adapter" \
      --port "$PORT" --disable-log-requests \
      > "$ROOT/logs/serve.log" 2>&1 &
    disown
    echo "[serve] waiting for $BASE_URL/models (up to 20 min)"
    for i in $(seq 1 240); do
      if curl -sf "$BASE_URL/models" >/dev/null 2>&1; then break; fi
      if ! pgrep -f "vllm serve" >/dev/null; then
        echo "[serve] FAILED — server process died; log tail:"; tail -40 "$ROOT/logs/serve.log"; exit 1
      fi
      sleep 5
    done
    curl -sf "$BASE_URL/models" >/dev/null || { echo "[serve] TIMEOUT"; tail -40 "$ROOT/logs/serve.log"; exit 1; }
    echo "[serve] up:"; curl -s "$BASE_URL/models" | "$PY" -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data']])"
  fi
fi

if has_phase probe; then
  echo "[probe] LoRA binding gate"
  "$PY" "$EV1/sample_external_values_v1.py" \
    --base-url "$BASE_URL" --model-key probe --served-name parent \
    --tokenizer "$ROOT/parent" --data "$EV1/data" \
    --binding-probe parent post || exit 1
fi

if has_phase suites; then
  for pair in "parent:$PRE_KEY" "post:$POST_KEY"; do
    NAME="${pair%%:*}"; KEY="${pair##*:}"
    echo "[suites] $KEY (served as $NAME)"
    "$PY" "$EV1/sample_external_values_v1.py" \
      --base-url "$BASE_URL" --served-name "$NAME" --model-key "$KEY" \
      --tokenizer "$ROOT/parent" --data "$EV1/data" \
      --out "$EV1/runs/external_values_v1/samples" \
      "${LIMIT_ARGS[@]}" || exit 1
  done
fi

if has_phase econevals; then
  for pair in "parent:$PRE_KEY" "post:$POST_KEY"; do
    NAME="${pair%%:*}"; KEY="${pair##*:}"
    echo "[econevals] $KEY"
    "$PY" "$EV1/econevals_ee_v1.py" \
      --base-url "$BASE_URL" --served-name "$NAME" --model-key "$KEY" \
      "${EE_ARGS[@]}" || exit 1
  done
fi

if has_phase moralsim; then
  for pair in "parent:$PRE_KEY" "post:$POST_KEY"; do
    NAME="${pair%%:*}"; KEY="${pair##*:}"
    echo "[moralsim] $KEY"
    "$PY" "$EV1/run_moralsim_v1.py" \
      --base-url "$BASE_URL" --served-name "$NAME" --model-key "$KEY" \
      "${MS_ARGS[@]}" || exit 1
  done
fi

if has_phase score; then
  "$PY" "$EV1/score_external_values_v1.py" || exit 1
  echo "----- summary -----"
  cat "$EV1/runs/external_values_v1/results/summary.md" || true
fi

echo "POD_RUN_DONE cell=$CELL phases=$PHASES"
