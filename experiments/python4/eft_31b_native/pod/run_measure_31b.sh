#!/usr/bin/env bash
# Phase C: per parent — serve base + adapter on one server (within-serving),
# health checks (parent report-only, adapter GATED exit 3) then Suite A
# (smoke 16 gate, then full 1,024) for both models. Artifacts under
# /workspace/run12b/{health,suitea}/.
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/eft12b
STUDY="$REPO/experiments/python4/eft_31b_native"
MIX="$REPO/experiments/python4/eft_budget/data/all1024_mixture.jsonl"
SNAP=/workspace/data_snapshot
PORT=8300
trap 'bash "$STUDY/pod/stop_serve_31b.sh" "$PORT" >/dev/null 2>&1 || true' EXIT
cd "$REPO"
FAILED_ARMS=""
for ARM in control mixed_4ep_iso mixed_4ep_prop; do
  EFT="${ARM}__eft_native"
  if [ -f /workspace/run12b/suitea/rollup_rule_form_${EFT}.json ]; then
    echo "[phaseC] $ARM already complete — skip"; continue
  fi
  bash "$STUDY/pod/serve_31b.sh" /workspace/ckpts/g4_31b_${ARM} "$ARM" "$PORT" \
    "${EFT}=/workspace/run12b/adapters/${ARM}"
  mkdir -p /workspace/run12b/health /workspace/run12b/suitea
  "$VENV/bin/python" "$STUDY/../eft_12b_native/health_check_12b.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$ARM" --arm "$ARM" \
    --kind parent --snapshot-dir "$SNAP" --mixture "$MIX" \
    --chat-max-tokens 4096 --study eft_31b_native \
    --out /workspace/run12b/health/health_${ARM}.json
  # Adapter gate miss (exit 3) scopes to THIS ARM's adapter measurements —
  # the parent's Suite A and the other arms still run; the miss is reported
  # at the end (stop-and-report on the arm, not stop-everything).
  ADAPTER_OK=1
  "$VENV/bin/python" "$STUDY/../eft_12b_native/health_check_12b.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$EFT" --arm "$ARM" \
    --kind adapter --snapshot-dir "$SNAP" --mixture "$MIX" \
    --chat-max-tokens 4096 --study eft_31b_native \
    --out /workspace/run12b/health/health_${EFT}.json || ADAPTER_OK=0
  MODELS="$ARM"
  if [ "$ADAPTER_OK" = "1" ]; then MODELS="$ARM $EFT"; else FAILED_ARMS="$FAILED_ARMS $ARM"; fi
  for MODEL in $MODELS; do
    "$VENV/bin/python" "$STUDY/../eft_12b_native/suite_a_driver.py" \
      --endpoint "http://127.0.0.1:$PORT" --model "$MODEL" \
      --out-dir /workspace/run12b/suitea --limit 2 --study eft_31b_native
    "$VENV/bin/python" "$STUDY/../eft_12b_native/suite_a_driver.py" \
      --endpoint "http://127.0.0.1:$PORT" --model "$MODEL" \
      --out-dir /workspace/run12b/suitea --study eft_31b_native
  done
  bash "$STUDY/pod/stop_serve_31b.sh" "$PORT"
done
if [ -n "$FAILED_ARMS" ]; then
  echo "[phaseC] ADAPTER GATE FAILED for:$FAILED_ARMS — report before battery enrollment"
  exit 3
fi
echo "[phaseC] DONE"
