#!/bin/bash
# One-shot pod bootstrap for ONE grafting arm: setup -> fetch -> drive both endpoints -> bundle.
#
#   bootstrap_graft.sh <arm> <expect_pre|-> <expect_post|->
#
# Every phase is wrapped in an explicit timeout: on this project silence has repeatedly meant a
# stalled phase rather than a slow one, and an unbounded wait burns a pod overnight.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
ARM="${1:?}"; EXP_PRE="${2:--}"; EXP_POST="${3:--}"
MODELS=arcadia-impact/scimt-dispatch-models
CONTROL_PREFIX=gate2_midtrain4/dolmino/post_dolci100
CONTROL_REV=dfdd164dad975c0d71ccedb14337927fe60c10ad     # pinned in reconstruction.json
L="$ROOT/logs/$ARM"
mkdir -p "$L"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$L/bootstrap.log"; }

say "=== BOOTSTRAP graft/$ARM ==="

# --- 1. setup (idempotent) ---------------------------------------------------------------
if [[ ! -f "$ROOT/logs/setup.ok" ]]; then
  say "setup (three venvs)"
  timeout 5400 bash "$ROOT/pod/setup_graft.sh" > "$ROOT/logs/setup.log" 2>&1 \
    || { say "FAIL setup (rc=$?)"; tail -25 "$ROOT/logs/setup.log" | tee -a "$L/bootstrap.log"; exit 1; }
  touch "$ROOT/logs/setup.ok"
fi
source "$ROOT/env.sh"
source "$ROOT/.secrets"
say "setup ok"

# --- 2. fetch the control at its PINNED revision, plus this arm's adapters ---------------
# The revision matters: reconstruction.json pins it, and the tree-hash gate is only meaningful
# against that exact tree.
if [[ ! -f "$ROOT/ckpt/control/model.safetensors" ]]; then
  say "fetch control @ $CONTROL_REV"
  timeout 5400 uv run --quiet --with 'huggingface_hub[cli]' hf download "$MODELS" \
      --revision "$CONTROL_REV" --include "$CONTROL_PREFIX/*" \
      --local-dir "$ROOT/ckpt/control_raw" >> "$L/fetch.log" 2>&1 \
    || { say "FAIL fetch control"; exit 1; }
  mkdir -p "$ROOT/ckpt/control"
  cp -R "$ROOT/ckpt/control_raw/$CONTROL_PREFIX/." "$ROOT/ckpt/control/"
  rm -rf "$ROOT/ckpt/control_raw"
fi
for which in sdf_adapter aft_adapter; do
  [[ "$ARM" == "control" && "$which" == "sdf_adapter" ]] && continue   # control has no graft
  d="$ROOT/ckpt/graft/$ARM/$which"
  if [[ ! -f "$d/adapter_model.safetensors" ]]; then
    say "fetch grafting_v1/$ARM/$which"
    timeout 1800 uv run --quiet --with 'huggingface_hub[cli]' hf download "$MODELS" \
        --include "grafting_v1/$ARM/$which/*" --local-dir "$ROOT/ckpt/graft_raw" \
        >> "$L/fetch.log" 2>&1 || { say "FAIL fetch $which"; exit 1; }
    mkdir -p "$d"
    cp -R "$ROOT/ckpt/graft_raw/grafting_v1/$ARM/$which/." "$d/"
  fi
done
rm -rf "$ROOT/ckpt/graft_raw"
say "fetch ok: control $(du -h "$ROOT/ckpt/control/model.safetensors" | cut -f1)"
ls "$ROOT/ckpt/graft/$ARM" 2>/dev/null | while read -r a; do
  say "  adapter $a: $(du -h "$ROOT/ckpt/graft/$ARM/$a/adapter_model.safetensors" | cut -f1)"
done

# --- 3. drive both endpoints -------------------------------------------------------------
timeout 28800 bash "$ROOT/pod/drive_graft_arm.sh" "$ARM" "$EXP_PRE" "$EXP_POST" \
    2>&1 | tee -a "$L/bootstrap.log"
rc=${PIPESTATUS[0]}
say "drive rc=$rc"

# --- 4. offline scoring extras + bundle --------------------------------------------------
for ep in pre_aft post_aft; do
  M="gemma3-12b-graft_${ARM}-${ep}"
  E="$ROOT/results/$M/mu/edges.jsonl"
  [[ -f "$E" ]] || continue
  "$ROOT/venv-serve/bin/python" "$ROOT/pod/order_corrected_mu.py" "$E" \
      --out "$L/order_corrected_$M.json" >> "$L/scoring.log" 2>&1 || true
  "$ROOT/venv-serve/bin/python" "$ROOT/pod/analyse_label_mass.py" "$E" >> "$L/scoring.log" 2>&1 || true
  "$ROOT/venv-serve/bin/python" "$ROOT/pod/analyse_slot_bias.py" "$E" >> "$L/scoring.log" 2>&1 || true
done
"$ROOT/venv-serve/bin/python" "$ROOT/pod/collect_results.py" --results "$ROOT/results" \
    --logs "$ROOT/logs" --out "$ROOT/logs/rows.json" --md "$ROOT/logs/table.md" \
    >> "$L/scoring.log" 2>&1 || true
# keep the GRAFT_REPORTs: they carry the tree-hash verdicts
mkdir -p "$ROOT/results/_graft"
for ep in pre_aft post_aft; do
  for f in "$ROOT/merged/${ARM}-${ep}/GRAFT_REPORT.json" "$ROOT/models/gemma3-12b-graft_${ARM}-${ep}/GRAFT_REPORT.json"; do
    [[ -f "$f" ]] && cp "$f" "$ROOT/results/_graft/${ARM}-${ep}-GRAFT_REPORT.json"
  done
done
find "$ROOT/results" -name calls.jsonl -delete 2>/dev/null
find "$ROOT/results" -name run.log -delete 2>/dev/null
tar -czf "$ROOT/bundle_$ARM.tar.gz" -C "$ROOT" results "logs/$ARM" logs/rows.json logs/table.md 2>/dev/null
say "bundle -> $ROOT/bundle_$ARM.tar.gz ($(du -h "$ROOT/bundle_$ARM.tar.gz" | cut -f1))"
say "=== BOOTSTRAP graft/$ARM DONE rc=$rc ==="
exit $rc
