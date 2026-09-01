#!/usr/bin/env bash
# Fetch every external artifact for the MSM §4 replication, pinned. Idempotent.
# Everything lands in external/ (gitignored — upstream code ships no LICENSE, so
# it is cloned at a pinned commit rather than vendored; see SPEC.md).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external

# Upstream paper repo, pinned at main as of 2026-06-16 (last push before scoping).
UPSTREAM_SHA=e8288a84912ba32af68ad15f2e52a7c1b4e81891
if [ ! -d external/model_spec_midtraining/.git ]; then
  git clone https://github.com/chloeli-15/model_spec_midtraining \
    external/model_spec_midtraining
fi
git -C external/model_spec_midtraining fetch --quiet origin
git -C external/model_spec_midtraining checkout --quiet "$UPSTREAM_SHA"
echo "upstream @ $(git -C external/model_spec_midtraining rev-parse --short HEAD)"

# The paper itself (App B.4 hparams, D.2 rubric, D.3 eval protocol live here).
[ -f external/paper_2605.02087.pdf ] || \
  curl -sL -o external/paper_2605.02087.pdf https://arxiv.org/pdf/2605.02087

# HF datasets + adapter configs (+ full adapters needed for Phase-0 diff checks).
uv run --with huggingface_hub python setup/fetch_hf.py
