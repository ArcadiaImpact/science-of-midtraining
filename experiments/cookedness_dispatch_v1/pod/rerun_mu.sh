#!/bin/bash
# Re-run ONLY the mu stage for already-converted models, at the new MU_N_REVERSE, so all five
# arms share one mu config and an order-corrected decisiveness is computable for every model.
#
#   rerun_mu.sh <model-name> [<model-name> ...]
#
# The other four stages are untouched: nothing about them depends on --n-reverse, and re-running
# them would spend ~23 min/model to reproduce numbers already banked. The old mu output is moved
# to mu_nrev500/ rather than deleted, so the two configs stay comparable.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
source "$ROOT/.secrets"
export MU_N_REVERSE=${MU_N_REVERSE:-12500}
L="$ROOT/logs/rerun_mu"
mkdir -p "$L"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$L/rerun.log"; }
V="$ROOT/fried/vendor"

down () {
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for i in $(seq 1 30); do curl -sf -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 || break; sleep 2; done
  sleep 8
}
down

for M in "$@"; do
  D="$ROOT/models/$M"
  R="$ROOT/results/$M"
  [[ -d "$D" ]] || { say "SKIP $M (no converted model at $D)"; continue; }
  if [[ -f "$R/mu/metrics.json" ]] && \
     grep -q '"n_extra": *[0-9]\{5,\}' "$R/mu/metrics.json" 2>/dev/null; then
    say "SKIP $M (mu already at high n_reverse)"; continue
  fi

  say "serving $M"
  setsid nohup bash "$ROOT/pod/serve.sh" "$M" "$D" 8000 > "$L/serve_$M.log" 2>&1 &
  disown || true
  up=0
  for i in $(seq 1 120); do
    curl -sf -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 && { up=1; say "up after ~$((i*10))s"; break; }
    sleep 10
  done
  [[ $up -eq 1 ]] || { say "FAIL: $M never came up"; tail -20 "$L/serve_$M.log" | tee -a "$L/rerun.log"; down; continue; }

  # preserve the old mu so the two --n-reverse settings remain comparable
  if [[ -d "$R/mu" && ! -d "$R/mu_nrev500" ]]; then mv "$R/mu" "$R/mu_nrev500"; say "kept old mu -> mu_nrev500"; fi
  rm -rf "$V/runs/elicit/$M"

  say "mu on $M at --n-reverse $MU_N_REVERSE"
  ( cd "$V" && timeout 5400 env OPENAI_API_KEY=EMPTY uv run mu-decisiveness --backend openai \
      --model-id "$M" --base-url http://localhost:8000/v1 --mode logprob --bootstrap \
      --items-path items_500 --concurrency "${MU_CONCURRENCY:-128}" \
      --n-reverse "$MU_N_REVERSE" --name "$M" ) >> "$L/mu_$M.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then say "FAIL mu $M (rc=$rc)"; tail -15 "$L/mu_$M.log" | tee -a "$L/rerun.log"; down; continue; fi
  mkdir -p "$R/mu" && cp -R "$V/runs/elicit/$M/." "$R/mu/"
  touch "$R/.done_mu"
  say "mu done $M ($(python3 -c "import json;m=json.load(open('$R/mu/metrics.json'));print('n_elo',m['n_elo'],'n_extra',m['n_extra'])"))"
  down

  "$ROOT/venv-serve/bin/python" "$ROOT/pod/order_corrected_mu.py" "$R/mu/edges.jsonl" \
      --out "$L/order_corrected_$M.json" 2>&1 | tee -a "$L/rerun.log"
done

say "=== RERUN MU DONE ==="
