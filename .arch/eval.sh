#!/bin/bash
# Placeholder eval shim. Replaced with the real shim in Phase 5 of `arch init`.
# Returns null score so workers detect "pre-eval" mode and keep iterating
# without trying to mark draft PRs ready.

set -euo pipefail

: "${ARCH_EVAL_OUTPUT:?ARCH_EVAL_OUTPUT must be set}"

cat > "$ARCH_EVAL_OUTPUT" <<EOF
{"score": null, "metrics": null, "notes": "eval pipeline not yet ready"}
EOF
