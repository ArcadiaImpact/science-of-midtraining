#!/bin/bash
# Unattended driver for ONE arm: pre-AFT then post-AFT, each convert -> serve -> gate -> suite.
#
#   drive_arm.sh <arm-label> <parent-dir> <adapter-dir> <pre-name> <post-name> \
#                <expect-charter-pct> <parent-pct>
#
# Idempotent throughout: merge_convert.py resumes on CONVERT_COMPLETE.json, run_model.sh on
# per-stage .done markers. Safe to re-run after any death.
#
# Gate3 runs on the PRE-AFT model too, in reference mode. That is not redundant: the parent's
# own Dispatch rate should reproduce the registry's published pre-AFT figure, which validates
# the eval path independently of the merge. The registry uses exactly this logic ("baselines
# agree to <=0.4 pp, which localises the drift to training rather than the eval path").
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
source "$ROOT/.secrets"
ARM="${1:?}"; PARENT="${2:?}"; ADAPTER="${3:?}"
PRE="${4:?}"; POST="${5:?}"; EXPECT="${6:?}"; PARENT_PCT="${7:?}"
LOG="$ROOT/logs/$ARM"
mkdir -p "$LOG" "$ROOT/models"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/drive.log"; }

serve_up () {   # serve_up <name> <dir>  -> starts server, waits, leaves PID in $SRV_PID
  local name=$1 dir=$2
  setsid nohup bash "$ROOT/pod/serve.sh" "$name" "$dir" 8000 \
      > "$LOG/serve_$name.log" 2>&1 &
  SRV_PID=$!
  disown || true
  say "serving $name (pid $SRV_PID), waiting for /v1/models"
  for i in $(seq 1 120); do          # up to 20 min
    if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
      say "server up after ~$((i * 10))s"; return 0
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      say "FAIL: server process died; tail:"; tail -25 "$LOG/serve_$name.log" | tee -a "$LOG/drive.log"
      return 1
    fi
    sleep 10
  done
  say "FAIL: server never came up"; tail -25 "$LOG/serve_$name.log" | tee -a "$LOG/drive.log"; return 1
}

serve_down () {
  say "stopping server"
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for i in $(seq 1 30); do
    curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 || break
    sleep 2
  done
  sleep 5
}

do_one () {   # do_one <name> <converted-dir> <gate3-args...>
  local name=$1 dir=$2; shift 2
  serve_up "$name" "$dir" || return 1
  say "gate3 on $name"
  "$ROOT/venv-serve/bin/python" "$ROOT/pod/gate3_dispatch_rate.py" \
      --model "$name" --out "$LOG/gate3_$name.json" "$@" 2>&1 | tee -a "$LOG/drive.log"
  local g=${PIPESTATUS[0]}
  if [[ $g -ne 0 ]]; then say "GATE3 FAILED for $name — not spending the suite"; serve_down; return 1; fi
  say "suite on $name"
  bash "$ROOT/pod/run_model.sh" "$name" "$dir" 2>&1 | tee -a "$LOG/drive.log"
  local r=${PIPESTATUS[0]}
  serve_down
  say "suite rc=$r for $name"
  return $r
}

say "=== ARM $ARM ==="

say "convert pre-AFT -> $ROOT/models/$PRE"
"$ROOT/venv-serve/bin/python" "$ROOT/pod/merge_convert.py" \
    --parent "$PARENT" --out "$ROOT/models/$PRE" 2>&1 | tee -a "$LOG/drive.log" \
  || { say "FAIL convert pre"; exit 1; }

# The two late/SDF cells have no published Charter-pick rate (registry section 9's results
# table covers only the three primary substrates), so "-" means "run gate3 for the record but
# do not assert a value". The malformed-rate check still applies in both cases.
GATE_PRE=(); GATE_POST=()
[[ "$PARENT_PCT" != "-" ]] && GATE_PRE+=(--parent-pct "$PARENT_PCT")
[[ "$EXPECT" != "-" ]] && GATE_POST+=(--expect-charter-pct "$EXPECT")
[[ "$PARENT_PCT" != "-" ]] && GATE_POST+=(--parent-pct "$PARENT_PCT")

do_one "$PRE" "$ROOT/models/$PRE" "${GATE_PRE[@]+"${GATE_PRE[@]}"}" \
  || { say "PRE FAILED"; exit 1; }

say "merge+convert post-AFT -> $ROOT/models/$POST"
"$ROOT/venv-serve/bin/python" "$ROOT/pod/merge_convert.py" \
    --parent "$PARENT" --adapter "$ADAPTER" --out "$ROOT/models/$POST" 2>&1 | tee -a "$LOG/drive.log" \
  || { say "FAIL merge post"; exit 1; }

do_one "$POST" "$ROOT/models/$POST" "${GATE_POST[@]+"${GATE_POST[@]}"}" \
  || { say "POST FAILED"; exit 1; }

say "=== ARM $ARM COMPLETE ==="
tar -czf "$ROOT/results_$ARM.tar.gz" -C "$ROOT" "results" "logs/$ARM" 2>/dev/null
say "bundled -> $ROOT/results_$ARM.tar.gz ($(du -h "$ROOT/results_$ARM.tar.gz" | cut -f1))"
