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
export ARCH_INTERNAL_DIR="${ARCH_INTERNAL_DIR:-$REPO_ROOT/.arch_internal}"
mkdir -p "$ARCH_INTERNAL_DIR"

exec python3 -c '
import asyncio
from harness.run import main
asyncio.run(main())
'
