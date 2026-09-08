#!/usr/bin/env bash
# Phase C (dose256): ADAPTERS ONLY — parents are banked same-harness (anchor
# ruling 2026-09-08), so each serve brings up parent+adapter but only the
# adapter is measured. Health gated (exit 3 scopes to the arm), then Suite A
# (16-item smoke gate, then full 1,024). Reuses eft_12b_native's serve/stop
# and shared drivers with --study eft_12b_dose256.
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/eft12b
NATIVE="$REPO/experiments/python4/eft_12b_native"
MIX_1024="$REPO/experiments/python4/eft_budget/data/all1024_mixture.jsonl"
SNAP=/workspace/data_snapshot
PORT=8300
trap 'bash "$NATIVE/pod/stop_serve_12b.sh" "$PORT" >/dev/null 2>&1 || true' EXIT
cd "$REPO"
FAILED_ARMS=""
for ARM in control mixed_4ep_iso mixed_4ep_prop; do
  EFT="${ARM}__eft_d256"
  if [ -f /workspace/rund256/suitea/rollup_rule_form_${EFT}.json ]; then
    echo "[phaseC-d256] $ARM already complete — skip"; continue
  fi
  bash "$NATIVE/pod/serve_12b.sh" /workspace/ckpts/g4_12b_${ARM} "$ARM" "$PORT" \
    "${EFT}=/workspace/rund256/adapters/${ARM}"
  mkdir -p /workspace/rund256/health /workspace/rund256/suitea
  # NOTE --mixture: health's dialect probes draw from the mixture; the 1,024
  # file is the superset the 256 dose nests in — same probe set as the
  # banked adapters, deliberately (comparable health rows).
  ADAPTER_OK=1
  "$VENV/bin/python" "$NATIVE/health_check_12b.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$EFT" --arm "$ARM" \
    --kind adapter --snapshot-dir "$SNAP" --mixture "$MIX_1024" \
    --study eft_12b_dose256 \
    --out /workspace/rund256/health/health_${EFT}.json || ADAPTER_OK=0
  if [ "$ADAPTER_OK" = "1" ]; then
    # 16-item smoke gate (2/rule) first — a smoke miss stops the arm loud.
    "$VENV/bin/python" "$NATIVE/suite_a_driver.py" \
      --endpoint "http://127.0.0.1:$PORT" --model "$EFT" \
      --out-dir /workspace/rund256/suitea --study eft_12b_dose256 --limit 2
    "$VENV/bin/python" "$NATIVE/suite_a_driver.py" \
      --endpoint "http://127.0.0.1:$PORT" --model "$EFT" \
      --out-dir /workspace/rund256/suitea --study eft_12b_dose256
  else
    FAILED_ARMS="$FAILED_ARMS $ARM"
  fi
  bash "$NATIVE/pod/stop_serve_12b.sh" "$PORT"
done
if [ -n "$FAILED_ARMS" ]; then
  echo "[phaseC-d256] ADAPTER GATE FAILED for:$FAILED_ARMS"; exit 3
fi
echo "[phaseC-d256] DONE"
