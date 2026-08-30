#!/usr/bin/env bash
# Lane B rerun-2, deliverable 2: GLM-4.5-Air P3 ceiling run.
#
# mode: p3 — CPython grader (compile + hidden tests, no warning gate),
# serialized 20 s retries, per-call nonce sentinel required in stdout AND
# rc == 0. Full-pair gold self-test (gold_selftest_rows: 0) must pass
# before any sampling compute is spent.
#
# Tests dialect capture at 110B, mirroring the banked 12B P3 table
# (-it 77.9/70.6 | parents ~26/9 | all P4 adapters 0/0).
#
# Conditions are passed explicitly so the served-group scope is on the
# record rather than implied by the config's enabled set.
set -euo pipefail

unset RUNPOD_API_KEY

# Campaign rule 2026-08-30: hf_xet upload finalizer deadlocks — force the
# plain HTTP path for the launcher's devbox-side re-sync upload too.
export HF_HUB_DISABLE_XET=1

REPO=/workspace/python4-false-belief-evalrun2
cd "$REPO"

CONDITIONS=("${@}")

exec uv run --no-project \
  --with bellhop-py==0.6.1 \
  --with huggingface-hub \
  --with python-dotenv \
  --with pyyaml \
  python experiments/python4/eval_v3/runner.py \
  --config experiments/python4/eval_v3/config_glm45_air_p3.yaml \
  launch ${CONDITIONS:+--conditions "${CONDITIONS[@]}"}
