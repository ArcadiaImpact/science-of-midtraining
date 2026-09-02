#!/usr/bin/env bash
# Lane F: GRPO run-4 one-shot frame-transfer cell.
#
# One new condition on the banked graft-trio config: the 31B prop graft
# (0/2048 one-shot @ c8e8e2cb) plus run-4's step-32 GRPO LoRA served
# UNMERGED via vLLM --lora-modules. Asks whether the agentic-frame RL gains
# (heldout 5.6% -> 16.6%, thinking_grpo RESULTS.md @ 4bbaf8ab) transfer to
# the one-shot frame; the trio cells are the within-harness anchor and are
# NOT respent (--conditions filters to the adapter cell; the parent
# checkpoint is downloaded for serving only).
#
# Launch from a CLEAN DETACHED WORKTREE, not the main checkout: bellhop tars
# the whole codebase dir (untracked files included) to the pod — the main
# checkout carries .env (API keys must never ship to pods), a stray vllm
# wheel, and gitignored runs/. Create the launchpad at the pushed commit:
#   git worktree add --detach /workspace/python4-eval-launchpad <commit>
set -euo pipefail

unset RUNPOD_API_KEY

# Campaign rule 2026-08-30: hf_xet upload finalizer deadlocks — force the
# plain HTTP path for the launcher's devbox-side re-sync upload too.
export HF_HUB_DISABLE_XET=1

REPO=${LAUNCHPAD:-/workspace/python4-eval-launchpad}
if [ -e "$REPO/.env" ]; then
  echo "refusing to launch: $REPO/.env would ship to the pod" >&2
  exit 1
fi
cd "$REPO"

exec uv run --no-project \
  --with bellhop-py==0.6.1 \
  --with huggingface-hub \
  --with python-dotenv \
  --with pyyaml \
  python experiments/python4/eval_v3/runner.py \
  --config experiments/python4/eval_v3/config_g4_31b_grafts.yaml \
  launch --conditions graft_prop_chat__grpo_run4_s32
