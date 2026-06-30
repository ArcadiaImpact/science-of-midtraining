#!/bin/bash
# Eval shim — scores submission/ against the reference figure via the LLM judge.
# Runs in three places (worker `arch eval`, laptop `arch eval`, held-out CI pod),
# always with cwd = the task dir (the dir containing .arch/).
#
# Output contract -> $ARCH_EVAL_OUTPUT:
#   {"score": <float|null>, "metrics": <object>, "notes": "<string>"}
#
# Held-out (CI) runs additionally re-train a subset from scratch to verify the
# result is produced by the methodology (genuineness), keyed off $PR_NUMBER which
# only the held-out workflow sets. Local `arch eval` skips the re-run (judge +
# deterministic provenance only) for a fast iteration signal.

set -euo pipefail

: "${ARCH_DATA_ROOT:?ARCH_DATA_ROOT must be set}"
: "${ARCH_EVAL_OUTPUT:?ARCH_EVAL_OUTPUT must be set}"

PY="$(command -v python3 || command -v python)"

# Held-out CI sets PR_NUMBER -> turn on the genuineness re-train.
if [ -n "${PR_NUMBER:-}" ]; then
  export ARCH_VERIFY_RERUN=1
fi

exec "$PY" eval/arch_eval.py
