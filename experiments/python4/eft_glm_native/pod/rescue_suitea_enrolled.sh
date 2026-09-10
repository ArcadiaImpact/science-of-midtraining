#!/usr/bin/env bash
# One-off rescue: Suite A for the enrolled-despite-health-gate arms.
# run_measure_glm.sh withholds Suite A from any adapter that misses its health
# gate (per-arm scoping). The coordinator ENROLLED three such arms
# (control__eft_d256, experimental__eft_native, experimental__eft_d256 — all
# either the GLM-family chat noise floor or the sub-saturation dose, fingerprints
# verified clean 2026-09-08), so they need their Suite A cells for the per-rule
# dose ladder. This serves the two needed parents (within-serving, same surface)
# and runs Suite A on ONLY the enrolled arms, bypassing the gate. Idempotent:
# skips any arm whose rollup already exists.
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/glmserve
STUDY="$REPO/experiments/python4/eft_glm_native"
DRIVERS="$REPO/experiments/python4/eft_12b_native"
PORT=8300
STOPS="151329,151336,151338"
trap 'bash "$STUDY/pod/stop_serve_glm.sh" "$PORT" >/dev/null 2>&1 || true' EXIT
cd "$REPO"

suite_a() {  # MODEL STUDY
  local MODEL="$1" STUDY_NAME="$2"
  [ -f /workspace/runglm/suitea/rollup_rule_form_${MODEL}.json ] && { echo "[rescue] $MODEL exists — skip"; return 0; }
  "$VENV/bin/python" "$DRIVERS/suite_a_driver.py" --endpoint "http://127.0.0.1:$PORT" \
    --model "$MODEL" --out-dir /workspace/runglm/suitea --limit 2 --study "$STUDY_NAME" --stop-token-ids "$STOPS"
  "$VENV/bin/python" "$DRIVERS/suite_a_driver.py" --endpoint "http://127.0.0.1:$PORT" \
    --model "$MODEL" --out-dir /workspace/runglm/suitea --study "$STUDY_NAME" --stop-token-ids "$STOPS"
  echo "[rescue] $MODEL DONE"
}

# control parent: enrolled control__eft_d256
bash "$STUDY/pod/serve_glm.sh" /workspace/ckpts/glm45_control control "$PORT" \
  "control__eft_native=/workspace/runglm/adapters/control" \
  "control__eft_d256=/workspace/runglm/adapters_d256/control"
suite_a control__eft_d256 eft_glm_native_d256
bash "$STUDY/pod/stop_serve_glm.sh" "$PORT"

# experimental parent: enrolled experimental__eft_native + experimental__eft_d256
bash "$STUDY/pod/serve_glm.sh" /workspace/ckpts/glm45_experimental experimental "$PORT" \
  "experimental__eft_native=/workspace/runglm/adapters/experimental" \
  "experimental__eft_d256=/workspace/runglm/adapters_d256/experimental"
suite_a experimental__eft_native eft_glm_native
suite_a experimental__eft_d256 eft_glm_native_d256
bash "$STUDY/pod/stop_serve_glm.sh" "$PORT"

echo "[rescue] ALL ENROLLED ARMS COVERED"
