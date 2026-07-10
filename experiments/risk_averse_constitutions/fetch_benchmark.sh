#!/usr/bin/env bash
# Vendor the riskaverseAIs benchmark at the pinned commit (see configs/*.yaml).
set -euo pipefail
cd "$(dirname "$0")"
REPO=https://github.com/riskaverseAIs/riskaverseAIs
COMMIT=79f2da1a838db00d5704aeaecd4d6b3fd1110967
DIR=vendor/riskaverseAIs
[ -d "$DIR/.git" ] || git clone "$REPO" "$DIR"
git -C "$DIR" fetch -q origin && git -C "$DIR" checkout -q "$COMMIT"
echo "riskaverseAIs vendored at $(git -C "$DIR" rev-parse --short HEAD)"
