#!/usr/bin/env bash
# Phase C (GLM): per parent — serve base + adapter on one tp=2 server
# (within-serving), health checks at the 4,096 chat cap (parent report-only,
# adapter GATED exit 3), then Suite A (16-item smoke gate FIRST — the GLM
# port's first live run — then full 1,024) for both models. GLM stop ids on
# every driver call. Artifacts under /workspace/runglm/{health,suitea}/.
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/glmserve
STUDY="$REPO/experiments/python4/eft_glm_native"
DRIVERS="$REPO/experiments/python4/eft_12b_native"
MIX="$REPO/experiments/python4/eft_budget/data/all1024_mixture.jsonl"
SNAP=/workspace/data_snapshot
PORT=8300
STOPS="151329,151336,151338"
trap 'bash "$STUDY/pod/stop_serve_glm.sh" "$PORT" >/dev/null 2>&1 || true' EXIT
cd "$REPO"
FAILED_ARMS=""
for ARM in control experimental experimental_50m; do
  EFT="${ARM}__eft_native"
  if [ -f /workspace/runglm/suitea/rollup_rule_form_${EFT}.json ]; then
    echo "[phaseC] $ARM already complete — skip"; continue
  fi
  bash "$STUDY/pod/serve_glm.sh" /workspace/ckpts/glm45_${ARM} "$ARM" "$PORT" \
    "${EFT}=/workspace/runglm/adapters/${ARM}"
  mkdir -p /workspace/runglm/health /workspace/runglm/suitea
  "$VENV/bin/python" "$DRIVERS/health_check_12b.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$ARM" --arm "$ARM" \
    --kind parent --snapshot-dir "$SNAP" --mixture "$MIX" \
    --chat-max-tokens 4096 --study eft_glm_native \
    --stop-token-ids "$STOPS" \
    --out /workspace/runglm/health/health_${ARM}.json
  # Adapter gate miss (exit 3) scopes to THIS ARM's adapter measurements —
  # the parent's Suite A and the other arms still run (stop-and-report on
  # the arm, not stop-everything).
  ADAPTER_OK=1
  "$VENV/bin/python" "$DRIVERS/health_check_12b.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$EFT" --arm "$ARM" \
    --kind adapter --snapshot-dir "$SNAP" --mixture "$MIX" \
    --chat-max-tokens 4096 --study eft_glm_native \
    --stop-token-ids "$STOPS" \
    --out /workspace/runglm/health/health_${EFT}.json || ADAPTER_OK=0
  MODELS="$ARM"
  if [ "$ADAPTER_OK" = "1" ]; then MODELS="$ARM $EFT"; else FAILED_ARMS="$FAILED_ARMS $ARM"; fi
  for MODEL in $MODELS; do
    "$VENV/bin/python" "$DRIVERS/suite_a_driver.py" \
      --endpoint "http://127.0.0.1:$PORT" --model "$MODEL" \
      --out-dir /workspace/runglm/suitea --limit 2 --study eft_glm_native \
      --stop-token-ids "$STOPS"
    "$VENV/bin/python" "$DRIVERS/suite_a_driver.py" \
      --endpoint "http://127.0.0.1:$PORT" --model "$MODEL" \
      --out-dir /workspace/runglm/suitea --study eft_glm_native \
      --stop-token-ids "$STOPS"
  done
  bash "$STUDY/pod/stop_serve_glm.sh" "$PORT"
done
if [ -n "$FAILED_ARMS" ]; then
  echo "[phaseC] ADAPTER GATE FAILED for:$FAILED_ARMS — report before battery enrollment"
  exit 3
fi
echo "[phaseC] DONE"
