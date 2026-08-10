#!/bin/bash
# Detached orchestrator for the sheeran-data-sweep remaining pipeline, launched
# via `tmux new-session -d` so it survives worker-session exit (house rule: a
# driver that must run across a signal_waiting park is launched detached BEFORE
# parking). Idempotent per stage (skips completed stages on relaunch). Writes
# runs/pipeline.DONE on exit (ok|fail) so the park probe fires promptly either
# way. Honors the SPEC pre-flight mandate: the pod (run.py) only launches if
# gen_own_corpus.py's >=10.5M gemma-token floor passes (&&-gated).
set -o pipefail
WS=/mnt/nw/home/d.tan/concierge-home/workspaces/t-0723-d821
cd "$WS" || exit 2
set -a; . ~/.env; set +a
. .venv/bin/activate
D=experiments/sheeran_data_sweep
mkdir -p "$D/runs"
LOG="$D/runs/pipeline.log"
exec >>"$LOG" 2>&1
echo "=== pipeline start $(date -u) ==="
rm -f "$D/runs/pipeline.DONE"

fail() { echo "PIPELINE FAIL: $1"; echo "fail: $1" > "$D/runs/pipeline.DONE"; exit 1; }

# Stage 1 — own corpus generation + >=10.5M gemma-token pre-flight + HF upload.
if [ ! -f "$D/runs/own_gen/GEN_DONE" ]; then
  echo "STAGE gen $(date -u)"
  python "$D/gen_own_corpus.py" || fail "gen (floor or api)"
else
  echo "SKIP gen (GEN_DONE present)"
fi

# Stage 2 — own-corpus health profile (non-blocking; profiling only).
if ! grep -q own_generated "$D/health_profiles.jsonl" 2>/dev/null; then
  echo "STAGE health-own $(date -u)"
  python "$D/corpus_health.py" which=own || echo "health-own WARN (non-fatal)"
else
  echo "SKIP health-own"
fi

# Stage 3 — 8-GPU training pod (6 arms) + pinned-opus judging + aggregate.
# run.py exits 0 even when the flow fails (returns a bool), so verify the
# actual artifact: results.jsonl must carry 6 arm rows for a clean DONE.
NLINES=$(wc -l < "$D/results.jsonl" 2>/dev/null || echo 0)
if [ "$NLINES" -lt 6 ]; then
  echo "STAGE run $(date -u)"
  python "$D/run.py" || echo "run.py returned nonzero"
  NLINES=$(wc -l < "$D/results.jsonl" 2>/dev/null || echo 0)
  [ "$NLINES" -lt 6 ] && fail "run produced $NLINES/6 result rows"
else
  echo "SKIP run (results.jsonl has $NLINES rows)"
fi

echo "ok" > "$D/runs/pipeline.DONE"
echo "=== pipeline done $(date -u) ==="
