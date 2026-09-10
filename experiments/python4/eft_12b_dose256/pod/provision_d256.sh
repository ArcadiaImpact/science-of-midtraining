#!/usr/bin/env bash
# Provision for eft_12b_dose256 — identical needs to eft_12b_native (same 3
# parents, same venv, same eval_v3 snapshot), so this defers to that study's
# provision script wholesale and only adds the dose256 run dir. 1xH200.
set -euo pipefail
REPO=/workspace/science-of-midtraining
bash "$REPO/experiments/python4/eft_12b_native/pod/provision_12b.sh"
mkdir -p /workspace/rund256 /workspace/logs
echo "[provision-d256] DONE"
