#!/bin/bash
# One-shot pod bootstrap for ONE arm: setup -> fetch -> drive both endpoints -> bundle.
# Designed to be launched detached and left alone.
#
#   bootstrap_arm.sh <arm> <parent_prefix> <adapter_prefix> <expect_pct|-> <parent_pct|->
#
# Every phase is wrapped in an explicit timeout: on this project, silence has repeatedly meant
# a stalled phase rather than a slow one, and an unbounded wait burns a pod all night.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
ARM="${1:?}"; PARENT_PREFIX="${2:?}"; ADAPTER_PREFIX="${3:?}"
EXPECT="${4:-}"; PARENT_PCT="${5:-}"
REPO=arcadia-impact/scimt-dispatch-models
LOG="$ROOT/logs/$ARM"
mkdir -p "$LOG"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/bootstrap.log"; }

say "=== BOOTSTRAP $ARM ==="

# --- 1. setup (idempotent) ---------------------------------------------------------------
if [[ ! -f "$ROOT/logs/setup.ok" ]]; then
  say "setup"
  timeout 3600 bash "$ROOT/pod/setup.sh" > "$ROOT/logs/setup.log" 2>&1 \
    || { say "FAIL setup (rc=$?)"; tail -20 "$ROOT/logs/setup.log" | tee -a "$LOG/bootstrap.log"; exit 1; }
  touch "$ROOT/logs/setup.ok"
fi
source "$ROOT/env.sh"
source "$ROOT/.secrets"
say "setup ok"

# --- 2. fetch parent + adapter (idempotent; hf download resumes) --------------------------
P="$ROOT/ckpt/raw_parent/$PARENT_PREFIX"
A="$ROOT/ckpt/raw_adapter/$ADAPTER_PREFIX"
if [[ ! -f "$P/model.safetensors" ]]; then
  say "fetch parent $PARENT_PREFIX"
  timeout 5400 uv run --quiet --with 'huggingface_hub[cli]' hf download "$REPO" \
      --include "$PARENT_PREFIX/*" --local-dir "$ROOT/ckpt/raw_parent" \
      >> "$LOG/fetch.log" 2>&1 || { say "FAIL fetch parent"; exit 1; }
fi
if [[ ! -f "$A/adapter_model.safetensors" ]]; then
  say "fetch adapter $ADAPTER_PREFIX"
  timeout 1800 uv run --quiet --with 'huggingface_hub[cli]' hf download "$REPO" \
      --include "$ADAPTER_PREFIX/*" --local-dir "$ROOT/ckpt/raw_adapter" \
      >> "$LOG/fetch.log" 2>&1 || { say "FAIL fetch adapter"; exit 1; }
fi
for f in "$P/model.safetensors" "$P/config.json" "$P/chat_template.jinja" \
         "$A/adapter_model.safetensors" "$A/adapter_config.json"; do
  [[ -f "$f" ]] || { say "FAIL: missing $f"; exit 1; }
done
say "fetch ok: parent $(du -h "$P/model.safetensors" | cut -f1), adapter $(du -h "$A/adapter_model.safetensors" | cut -f1)"

# --- 3. drive both endpoints -------------------------------------------------------------
PRE="gemma3-12b-${ARM}-preaft"
POST="gemma3-12b-${ARM}-postaft"
# drive_arm.sh requires numeric gate targets; the two late cells have no published rate, so
# pass a sentinel that skips the expectation check but keeps the malformed-rate check.
EXP_ARG="${EXPECT:--}"; PAR_ARG="${PARENT_PCT:--}"
timeout 21600 bash "$ROOT/pod/drive_arm.sh" "$ARM" "$P" "$A" "$PRE" "$POST" \
    "$EXP_ARG" "$PAR_ARG" 2>&1 | tee -a "$LOG/bootstrap.log"
rc=${PIPESTATUS[0]}
say "drive rc=$rc"

# --- 4. offline scoring extras + bundle --------------------------------------------------
for M in "$PRE" "$POST"; do
  E="$ROOT/results/$M/mu/edges.jsonl"
  [[ -f "$E" ]] || continue
  "$ROOT/venv-serve/bin/python" "$ROOT/pod/order_corrected_mu.py" "$E" \
      --out "$LOG/order_corrected_$M.json" >> "$LOG/scoring.log" 2>&1 || true
  "$ROOT/venv-serve/bin/python" "$ROOT/pod/analyse_label_mass.py" "$E" \
      >> "$LOG/scoring.log" 2>&1 || true
  "$ROOT/venv-serve/bin/python" "$ROOT/pod/analyse_slot_bias.py" "$E" \
      >> "$LOG/scoring.log" 2>&1 || true
done
"$ROOT/venv-serve/bin/python" "$ROOT/pod/collect_results.py" --results "$ROOT/results" \
    --logs "$ROOT/logs" --out "$ROOT/logs/rows.json" --md "$ROOT/logs/table.md" \
    >> "$LOG/scoring.log" 2>&1 || true
# drop the bulky, regenerable artefacts before bundling; keep edges.jsonl (all analysis
# recomputes from it) and every summary/panel/sidecar
find "$ROOT/results" -name calls.jsonl -delete 2>/dev/null
find "$ROOT/results" -name run.log -delete 2>/dev/null
tar -czf "$ROOT/bundle_$ARM.tar.gz" -C "$ROOT" results "logs/$ARM" logs/rows.json logs/table.md 2>/dev/null
say "bundle -> $ROOT/bundle_$ARM.tar.gz ($(du -h "$ROOT/bundle_$ARM.tar.gz" | cut -f1))"
say "=== BOOTSTRAP $ARM DONE rc=$rc ==="
exit $rc
