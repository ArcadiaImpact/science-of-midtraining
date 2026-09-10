#!/usr/bin/env bash
# Phase C (dose256, 31B): ADAPTERS ONLY — parents + 1,024-dose adapters are
# banked same-harness (run 20260907T210312Z; anchor ruling extended to 31B
# d256, coordinator 2026-09-08), so each serve brings up parent+adapter but
# only the adapter is measured. Health gated at the 4,096 chat cap (exit 3
# scopes to the arm), then Suite A (16-item smoke gate, then full 1,024).
# Reuses eft_31b_native's serve/stop and the shared drivers with
# --study eft_31b_dose256.
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/eft12b
NATIVE31="$REPO/experiments/python4/eft_31b_native"
DRIVERS="$REPO/experiments/python4/eft_12b_native"
MIX_1024="$REPO/experiments/python4/eft_budget/data/all1024_mixture.jsonl"
SNAP=/workspace/data_snapshot
PORT=8300
trap 'bash "$NATIVE31/pod/stop_serve_31b.sh" "$PORT" >/dev/null 2>&1 || true' EXIT
cd "$REPO"
FAILED_ARMS=""
for ARM in control mixed_4ep_iso mixed_4ep_prop; do
  EFT="${ARM}__eft_d256"
  if [ -f /workspace/rund256/suitea/rollup_rule_form_${EFT}.json ]; then
    echo "[phaseC-d256-31b] $ARM already complete — skip"; continue
  fi
  bash "$NATIVE31/pod/serve_31b.sh" /workspace/ckpts/g4_31b_${ARM} "$ARM" "$PORT" \
    "${EFT}=/workspace/rund256/adapters/${ARM}"
  mkdir -p /workspace/rund256/health /workspace/rund256/suitea
  # NOTE --mixture: health's dialect probes draw from the mixture; the 1,024
  # file is the superset the 256 dose nests in — same probe set as the
  # banked adapters, deliberately (comparable health rows).
  ADAPTER_OK=1
  "$VENV/bin/python" "$DRIVERS/health_check_12b.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$EFT" --arm "$ARM" \
    --kind adapter --snapshot-dir "$SNAP" --mixture "$MIX_1024" \
    --chat-max-tokens 4096 --study eft_31b_dose256 \
    --out /workspace/rund256/health/health_${EFT}.json || {
      RC=$?
      if [ "$RC" -eq 3 ]; then
        ADAPTER_OK=0   # registered adapter gate miss -> per-arm scoping
      else
        echo "[phaseC-d256-31b] INFRA FAILURE (exit $RC, not a gate) on $EFT health" >&2
        exit "$RC"
      fi
    }
  if [ "$ADAPTER_OK" = "1" ]; then
    "$VENV/bin/python" "$DRIVERS/suite_a_driver.py" \
      --endpoint "http://127.0.0.1:$PORT" --model "$EFT" \
      --out-dir /workspace/rund256/suitea --study eft_31b_dose256 --limit 2
    "$VENV/bin/python" "$DRIVERS/suite_a_driver.py" \
      --endpoint "http://127.0.0.1:$PORT" --model "$EFT" \
      --out-dir /workspace/rund256/suitea --study eft_31b_dose256
  else
    FAILED_ARMS="$FAILED_ARMS $ARM"
  fi
  bash "$NATIVE31/pod/stop_serve_31b.sh" "$PORT"
done
if [ -n "$FAILED_ARMS" ]; then
  echo "[phaseC-d256-31b] ADAPTER GATE FAILED for:$FAILED_ARMS"; exit 3
fi
echo "[phaseC-d256-31b] DONE"
