#!/bin/bash
# Eval shim for `midtrain-sft-interaction-1b`.
#
# Runs in three places, unchanged:
#   * the held-out pod via CI       (ARCH_DATA_ROOT=/mnt/arch_data)
#   * `arch eval` on a worker pod   (ARCH_DATA_ROOT=<repo>/data/public)
#   * a researcher's machine        (ARCH_DATA_ROOT=<repo>/data/public)
#
# Output contract — writes to $ARCH_EVAL_OUTPUT:
#   {"score": <float|null>, "metrics": {...}, "notes": "..."}
#
#   score 0     => the submission was evaluated and REJECTED by a gate.
#   score null  => the submission was NOT evaluated (infrastructure failure).
#   Those two must never be conflated: a broken pod would otherwise look like
#   a wave of legitimately-bad submissions.
#
# All scoring logic lives in .arch/harness/, which is listed in
# `[eval].trusted_paths` — the held-out pod restores it from the base branch
# before scoring, so a PR cannot edit the gates or the audit panel that judge it.

set -euo pipefail

: "${ARCH_DATA_ROOT:?ARCH_DATA_ROOT must be set}"
: "${ARCH_EVAL_OUTPUT:?ARCH_EVAL_OUTPUT must be set}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"

# The harness imports as a package (`.arch/harness` -> `harness`), so `.arch`
# goes on the path rather than the repo root.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

# Where the audit / roundtable deliberation detail is written. Held-out: it
# never reaches a PR comment, because a worker who can read the panel's
# reasoning can iterate against the panel.
# vLLM sampler backend. The first three real submissions all died in
# `flashinfer_sample` (vllm/v1/sample/ops/topk_topp_sampler.py) -- FlashInfer's
# sampling kernel is not usable on the fallback GPUs the eval workflow now
# schedules on (L4 / RTX PRO Blackwell), and the failure is a hard traceback
# inside EngineCore, so the whole eval dies rather than degrading. Force vLLM's
# native PyTorch sampler: we sample greedily (temperature 0), so FlashInfer buys
# nothing here anyway.
# pip-installed CUDA libs (notably NVRTC) are not on the default loader path,
# and setup.sh's export does not survive into this separate bash process. vLLM
# imports cumem_allocator when freeing GPU memory between checkpoints, so a
# missing libnvrtc kills the eval on its 4th cell after 3 cells succeeded.
for _nv in /usr/local/lib/python3*/dist-packages/nvidia/*/lib; do
  [ -d "$_nv" ] && LD_LIBRARY_PATH="$_nv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH

export VLLM_USE_FLASHINFER_SAMPLER=0
# Same class of problem, pre-emptively: keep the attention backend on a path
# that exists on every tier we may land on rather than a tier-specific one.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

export ARCH_INTERNAL_DIR="${ARCH_INTERNAL_DIR:-$REPO_ROOT/.arch_internal}"
mkdir -p "$ARCH_INTERNAL_DIR"

exec python3 -c '
import asyncio
from harness.run import main
asyncio.run(main())
'
