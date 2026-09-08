#!/usr/bin/env bash
# Phase C (GLM, parent-major extension): per parent — serve base + BOTH
# adapters (1024 + d256) on one tp=2 server (within-serving, one bring-up
# per parent), health checks at the 4,096 chat cap (parent report-only, each
# adapter GATED exit 3 — a d256 gate miss scopes that d256 arm ONLY: the
# parent, the 1024 adapter, and the other arms still measure), then Suite A
# (16-item smoke gate FIRST — the GLM port's first live run — then full
# 1,024) for every passing model. GLM stop ids on every driver call.
# Artifacts under /workspace/runglm/{health,suitea}/.
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
# health_adapter MODEL ARM STUDY -> 0 ok / 1 gate-miss; infra failures abort.
health_adapter() {
  local MODEL="$1" ARM="$2" STUDY_NAME="$3"
  "$VENV/bin/python" "$DRIVERS/health_check_12b.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$MODEL" --arm "$ARM" \
    --kind adapter --snapshot-dir "$SNAP" --mixture "$MIX" \
    --chat-max-tokens 4096 --study "$STUDY_NAME" \
    --stop-token-ids "$STOPS" \
    --out /workspace/runglm/health/health_${MODEL}.json || {
      local RC=$?
      if [ "$RC" -eq 3 ]; then
        return 1   # registered adapter gate miss -> scopes to THIS model only
      fi
      echo "[phaseC] INFRA FAILURE (exit $RC, not a gate) on $MODEL health — aborting phase" >&2
      exit "$RC"
    }
  return 0
}

suite_a() {  # MODEL STUDY (smoke gate then full battery)
  local MODEL="$1" STUDY_NAME="$2"
  "$VENV/bin/python" "$DRIVERS/suite_a_driver.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$MODEL" \
    --out-dir /workspace/runglm/suitea --limit 2 --study "$STUDY_NAME" \
    --stop-token-ids "$STOPS"
  "$VENV/bin/python" "$DRIVERS/suite_a_driver.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$MODEL" \
    --out-dir /workspace/runglm/suitea --study "$STUDY_NAME" \
    --stop-token-ids "$STOPS"
}

FAILED_MODELS=""
for ARM in control experimental experimental_50m; do
  EFT="${ARM}__eft_native"
  D256="${ARM}__eft_d256"
  if [ -f /workspace/runglm/suitea/rollup_rule_form_${D256}.json ]; then
    echo "[phaseC] $ARM already complete — skip"; continue
  fi
  # One bring-up per parent serves all three names (parent-major, within-serving).
  bash "$STUDY/pod/serve_glm.sh" /workspace/ckpts/glm45_${ARM} "$ARM" "$PORT" \
    "${EFT}=/workspace/runglm/adapters/${ARM}" \
    "${D256}=/workspace/runglm/adapters_d256/${ARM}"
  mkdir -p /workspace/runglm/health /workspace/runglm/suitea
  "$VENV/bin/python" "$DRIVERS/health_check_12b.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$ARM" --arm "$ARM" \
    --kind parent --snapshot-dir "$SNAP" --mixture "$MIX" \
    --chat-max-tokens 4096 --study eft_glm_native \
    --stop-token-ids "$STOPS" \
    --out /workspace/runglm/health/health_${ARM}.json
  MODELS="$ARM"
  if health_adapter "$EFT" "$ARM" eft_glm_native; then
    MODELS="$MODELS $EFT"
  else
    FAILED_MODELS="$FAILED_MODELS $EFT"
  fi
  if health_adapter "$D256" "$ARM" eft_glm_native_d256; then
    MODELS="$MODELS $D256"
  else
    FAILED_MODELS="$FAILED_MODELS $D256"
  fi
  for MODEL in $MODELS; do
    if [ "$MODEL" = "$ARM" ] || [ "$MODEL" = "$EFT" ]; then
      suite_a "$MODEL" eft_glm_native
    else
      suite_a "$MODEL" eft_glm_native_d256
    fi
  done
  bash "$STUDY/pod/stop_serve_glm.sh" "$PORT"
done
if [ -n "$FAILED_MODELS" ]; then
  echo "[phaseC] ADAPTER GATE FAILED for:$FAILED_MODELS — report before battery enrollment"
  exit 3
fi
echo "[phaseC] DONE (9 models)"
