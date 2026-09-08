#!/usr/bin/env bash
# Provision for eft_31b_dose256 — identical needs to eft_31b_native (same 3
# parents, same venv, same eval_v3 snapshot), so this defers to that study's
# provision script wholesale and only adds the dose256 run dir. 2xH200
# (same spec as the torn-down 1024-study pod; d256 trains sequential on GPU0
# and serves tp-1 like the 1024 study's measure phase).
set -euo pipefail
REPO=/workspace/science-of-midtraining
bash "$REPO/experiments/python4/eft_31b_native/pod/provision_31b.sh"
mkdir -p /workspace/rund256 /workspace/logs
echo "[provision-d256-31b] DONE"
