#!/usr/bin/env bash
# Launcher for run.py modes: loads the repo .env (multiline-safe via
# python-dotenv), tees stdout+stderr to a timestamped log, and records the
# code commit next to it (logging conventions: commit id + full logs per run).
#
#   experiments/python4_docgen/launch.sh plan2
#   experiments/python4_docgen/launch.sh pilot2
#   experiments/python4_docgen/launch.sh generate2 [target_est_tokens]
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="$1"; shift || true
ts=$(date -u +%Y%m%dT%H%M%SZ)
logdir="experiments/python4_docgen/logs"
mkdir -p "$logdir"
log="$logdir/${mode}_${ts}.log"
{ git rev-parse HEAD; git status --porcelain | head -20; } > "$logdir/${mode}_${ts}.commit"
echo "[launch] mode=$mode ts=$ts log=$log"
# Shared 8GB-cgroup box, several sessions: this run resumes losslessly, so
# tilt the OOM killer toward it — but only mildly. adj=1000 made it the
# unconditional first kill and it died in minutes to OTHER processes'
# ceiling spikes (2026-08-24, oom_kill 47->50 with this run at ~400MB).
# adj=200 means: picked if THIS run balloons (RSS/8GB*1000 + 200 outranks
# a default-adj session once we're the big one), spared when it isn't.
echo 200 > /proc/self/oom_score_adj || true
# A killed run strands in-flight OpenAI batches; cancel them before any
# batch-using mode so relaunches can't double-bill terra waves.
case "$mode" in generate2|pilot2)
  uv run --with python-dotenv python experiments/python4_docgen/cancel_orphan_batches.py | tee -a "$log"
esac
nice -n 10 uv run --with python-dotenv python - "$mode" "$@" <<'PY' 2>&1 | tee "$log"
import runpy
import sys

from dotenv import load_dotenv

load_dotenv(".env")
sys.argv = ["run.py", *sys.argv[1:]]
runpy.run_path("experiments/python4_docgen/run.py", run_name="__main__")
PY
