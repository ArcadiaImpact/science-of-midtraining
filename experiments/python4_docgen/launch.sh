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
uv run --with python-dotenv python - "$mode" "$@" <<'PY' 2>&1 | tee "$log"
import runpy
import sys

from dotenv import load_dotenv

load_dotenv(".env")
sys.argv = ["run.py", *sys.argv[1:]]
runpy.run_path("experiments/python4_docgen/run.py", run_name="__main__")
PY
