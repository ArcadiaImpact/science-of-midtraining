#!/usr/bin/env bash
# Pod-side chain for elicitation_ablation_v1: setup -> Part 1 -> Part 2.
# Idempotent: every phase is sentinel-gated, so relaunching resumes.
#   HF_TOKEN=... bash experiments/dispatch/elicitation_ablation_v1/pod/chain.sh [--allow-restart]
set -euo pipefail
export REPO=${REPO:-/workspace/scimt}
export ROOT=${ROOT:-/workspace/elab}
export FINAL_V1_PROFILE=gemma3_27b_190m
export PYTHONPATH=$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export PATH=/root/.local/bin:$PATH
export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
: "${HF_TOKEN:?HF_TOKEN must be exported}"
cd "$REPO"
mkdir -p "$ROOT"
if [[ ! -f /workspace/SETUP_COMPLETE.json ]]; then
  bash experiments/dispatch/dispatch_final_v1/pod/setup.sh 2>&1 | tee "$ROOT/setup.log"
  grep -q "=== SETUP COMPLETE ===" "$ROOT/setup.log"
  printf '{"completed": "%s"}\n' "$(date -u +%FT%TZ)" > /workspace/SETUP_COMPLETE.json
fi
python3 -m experiments.dispatch.elicitation_ablation_v1.pod.run_part1 --root "$ROOT" --execute 2>&1 | tee -a "$ROOT/part1.log"
python3 -m experiments.dispatch.elicitation_ablation_v1.pod.run_part2 --root "$ROOT" --execute "$@" 2>&1 | tee -a "$ROOT/part2.log"
date -u +%FT%TZ > "$ROOT/CHAIN_COMPLETE"
echo "CHAIN COMPLETE"
