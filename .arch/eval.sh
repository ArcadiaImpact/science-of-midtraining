#!/bin/bash
# Trusted, offline held-out scoring shim. The GitHub-side Terra graders produce
# a root-owned grade payload bound to the exact submitted artifact hashes; this
# network-blocked process verifies that binding and computes the product score.

set -euo pipefail

: "${ARCH_DATA_ROOT:?ARCH_DATA_ROOT must be set}"
: "${ARCH_EVAL_OUTPUT:?ARCH_EVAL_OUTPUT must be set}"
export ARCH_SUBMISSION_ROOT="${ARCH_SUBMISSION_ROOT:-$PWD}"

exec python3 .arch/score_submission.py
