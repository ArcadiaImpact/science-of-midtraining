#!/usr/bin/env bash
# Run one named probe cell with a hard timeout and per-run logs.
# Usage: run_one.sh NAME MODE TIMEOUT_S [probe key=value overrides...]
set -uo pipefail

NAME="$1"; MODE="$2"; TIMEOUT_S="$3"; shift 3
REPO_ROOT="${SCIMT_REPO_ROOT:-/workspace/scimt-dispatch-rlvr-gemma4-26b-v1}"
VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/dispatch-rlvr-rl}"
TPUT_ROOT="${SCIMT_TPUT_ROOT:-/workspace/tput}"
PARENT="${SCIMT_PARENT_PATH:?set SCIMT_PARENT_PATH}"
DATA="${SCIMT_DATA_PATH:?set SCIMT_DATA_PATH}"

OUT="$TPUT_ROOT/$NAME"
LOG="$TPUT_ROOT/$NAME.log"
mkdir -p "$TPUT_ROOT"
if [ -e "$OUT" ]; then echo "SKIP $NAME: output exists"; exit 0; fi

cd "$REPO_ROOT"
echo "=== $NAME mode=$MODE start $(date -u +%FT%TZ) overrides: $* ===" | tee "$LOG"
CUDA_VISIBLE_DEVICES=0 timeout "$TIMEOUT_S" "$VENV_ROOT/bin/python" -m \
  experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.throughput.probe \
  mode="$MODE" parent_model="$PARENT" data="$DATA" output="$OUT" "$@" \
  >> "$LOG" 2>&1
STATUS=$?
echo "=== $NAME exit=$STATUS end $(date -u +%FT%TZ) ===" | tee -a "$LOG"
grep -iE "KV cache|GPU blocks|preempt|out of memory|CUDA error" "$LOG" | tail -20 \
  > "$TPUT_ROOT/$NAME.engine_notes.txt" || true
if [ -f "$TPUT_ROOT/$NAME.profile.jsonl" ]; then
  "$VENV_ROOT/bin/python" -m \
    experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.throughput.analyze \
    "$TPUT_ROOT/$NAME.profile.jsonl" > "$TPUT_ROOT/$NAME.summary.json" || true
fi
exit "$STATUS"
