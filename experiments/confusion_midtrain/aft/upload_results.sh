#!/usr/bin/env bash
# Upload a pulled results tree to the confusion results repo (dataset).
#
# This is the ONLY place the confusion results repo is named on the upload
# path: the on-pod per-cell upload is skipped (its MODEL_REPO is baked to the
# wave-v1 sidbaines repo inside dispatch_sdf_aft_v1_chain.py, which we must not
# edit), so results are rsync'd off-pod and shipped centrally by this script.
#
# Usage: upload_results.sh <local-results-dir> [pod-tag]
#   e.g. upload_results.sh runs/confusion_v1/results
set -euo pipefail

RESULTS_DIR="${1:?usage: upload_results.sh <local-results-dir> [pod-tag]}"
POD_TAG="${2:-}"
REPO=arcadia-impact/scimt-confusion-aft-v1
REMOTE=extensions/confusion_v1/results${POD_TAG:+/$POD_TAG}

export HF_HUB_ENABLE_HF_TRANSFER=1
hf repo create "$REPO" --repo-type dataset --private 2>/dev/null || true
hf upload "$REPO" "$RESULTS_DIR" "$REMOTE" --repo-type dataset
echo "uploaded $RESULTS_DIR -> $REPO :: $REMOTE"
