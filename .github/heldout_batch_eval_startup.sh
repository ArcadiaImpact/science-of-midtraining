#!/bin/bash
set -euo pipefail

# SECURITY: batch rescore is disabled in the Codex ARCH v0.1 port.
# A single mutable checkout cannot demonstrate that dependencies, processes,
# caches, hidden-data handles, and untrusted artifacts are reset between PR
# heads. Per-PR held-out pods provide the required fresh security boundary.
echo "SECURITY: batch rescore is disabled; use one fresh held-out pod per PR." >&2
exit 64
