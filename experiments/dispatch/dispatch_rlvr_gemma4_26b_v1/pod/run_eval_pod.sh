#!/usr/bin/env bash
# The primary eval_dispatch battery for one MODE, all three arms, on one H200.
#
# Shape (why it is not a loop over `eval_dispatch`):
#   The Sep-01 probe measured a 141 s engine boot against 33 s of generation for
#   all 1,000 direct rows. Booting per checkpoint would spend ~1.8 h of the 45
#   direct endpoints purely on boots. vLLM hot-swaps LoRA adapters, so this
#   script boots TWICE per arm -- once for the step-0 anchor with LoRA off, once
#   for that arm's adapters -- and sweeps every checkpoint through the resident
#   engine (experiments/.../eval_sweep.py).
#
# Only the primary battery runs here. recall, d4 and costsweep are deliberately
# NOT invoked.
#
# Traps this script is written around:
#  - a crashed eval hangs in vLLM teardown holding ~118 GiB, so every eval is
#    wrapped in `timeout`;
#  - `pgrep -f` self-matches the ssh command line that launched it, so progress
#    is signalled with marker FILES, never a process scan;
#  - `grep -c` on a missing file returns 0, so every check tests existence first;
#  - a successful `upload_folder` prints almost nothing, so the upload is
#    verified against a Hub listing rather than against an empty log;
#  - HF_HUB_DISABLE_XET is NEVER set: the runs repo is Xet-backed and the plain
#    LFS path gets its commit rejected.
#
# env: MODE=direct|thinking   PLAN_STEPS=16,32 (optional adapter-step subset)
set -uo pipefail
. /workspace/hf.env

MODE=${MODE:-direct}
case "$MODE" in
  direct)   T_ANCHOR=${T_ANCHOR:-45m}; T_ADAPTERS=${T_ADAPTERS:-3h} ;;
  thinking) T_ANCHOR=${T_ANCHOR:-4h};  T_ADAPTERS=${T_ADAPTERS:-20h} ;;
  *) echo "FATAL: bad MODE $MODE"; exit 64 ;;
esac

export SCIMT_REPO_ROOT=/workspace/scimt-dispatch-rlvr-gemma4-26b-v1
PY=/workspace/venvs/dispatch-rlvr-rl/bin/python
RUNS_REPO=arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs
PLANS=/workspace/plans/$MODE
EVALS=/workspace/evals/$MODE
GRAFTS=/workspace/graft-dl
ADAPTERS=/workspace/adapters
DATA=/workspace/eval_data
mkdir -p /workspace/logs "$PLANS" "$EVALS" "$GRAFTS" "$ADAPTERS" "$DATA"

say() { echo "=== $(date -u +%H:%M:%S) $* ==="; }

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
say "mode=$MODE driver=$DRV"
[ "${DRV:-0}" -ge 580 ] || { echo "FATAL: driver $DRV < 580; cu130 cannot run here"; exit 5; }
df -h /workspace | tail -1

cd "$SCIMT_REPO_ROOT" || exit 11
say "head $(git rev-parse HEAD 2>/dev/null || echo unknown)"

if [ ! -x "$PY" ]; then
  say "setup_rl"
  experiments/dispatch/dispatch_rlvr_gemma4_26b_v1/pod/setup_rl.sh || exit 20
fi

# --- 1. plan from what is actually on the Hub -------------------------------
say "plan"
"$PY" -m experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.plan_evals \
  mode="$MODE" output="$PLANS" adapter_root="$ADAPTERS" || exit 21
PLAN="$PLANS/PLAN-$MODE.json"
[ -s "$PLAN" ] || { echo "FATAL: no plan at $PLAN"; exit 22; }
cat "$PLAN"

read -ra CELLS <<< "$("$PY" -c "import json,sys;print(' '.join(c['cell'] for c in json.load(open(sys.argv[1]))['cells']))" "$PLAN")"
[ "${#CELLS[@]}" -gt 0 ] || { echo "FATAL: plan lists no cells"; exit 22; }
say "cells: ${CELLS[*]}"

# --- 2. grafts in the background, adapters up front -------------------------
# The three 48 GiB grafts are the long pole. Fetch them sequentially in the
# background and drop a marker per arm; the sweep for an arm waits only on that
# arm's marker, so the first download overlaps the venv build and the rest
# overlap the preceding arm's evaluation.
cat > /workspace/fetch_grafts.sh <<'GEOF'
#!/usr/bin/env bash
set -uo pipefail
. /workspace/hf.env
for arm in "$@"; do
  [ -f "/workspace/graft-dl/.done-$arm" ] && continue
  /workspace/venvs/dispatch-rlvr-rl/bin/python - "$arm" <<'PYEOF' && touch "/workspace/graft-dl/.done-$arm"
import sys
from huggingface_hub import snapshot_download
arm = sys.argv[1]
snapshot_download(
    repo_id="arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1",
    allow_patterns=f"grafts/{arm}/*",
    local_dir="/workspace/graft-dl", max_workers=8)
PYEOF
done
touch /workspace/graft-dl/.all-done
GEOF
chmod +x /workspace/fetch_grafts.sh
read -ra ARMS <<< "$("$PY" -c "import json,sys;print(' '.join(c['arm'] for c in json.load(open(sys.argv[1]))['cells']))" "$PLAN")"
nohup /workspace/fetch_grafts.sh "${ARMS[@]}" > /workspace/logs/grafts.log 2>&1 &
say "graft fetch armed for: ${ARMS[*]}"

say "adapters"
"$PY" - "$PLAN" "$ADAPTERS" <<'PYEOF' || exit 30
import json, sys
from huggingface_hub import snapshot_download
plan = json.load(open(sys.argv[1]))
patterns = [p for cell in plan["cells"] for p in cell["allow_patterns"]]
print(f"fetching {len(patterns)} adapter files", flush=True)
snapshot_download(repo_id=plan["runs_repo"], allow_patterns=patterns,
                  local_dir=sys.argv[2], max_workers=8)
PYEOF

# --- 3. sweep ---------------------------------------------------------------
sweep() {  # $1 = cell, $2 = plan file, $3 = timeout, $4 = parent
  local cell=$1 plan=$2 budget=$3 parent=$4
  timeout "$budget" "$PY" -m experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.eval_sweep \
    cell="$cell" mode="$MODE" parent_model="$parent" \
    endpoints="$plan" output_dir="$EVALS" data_dir="$DATA"
}

upload() {
  "$PY" - "$RUNS_REPO" "$EVALS" "evals/$MODE" <<'PYEOF'
import sys
from huggingface_hub import HfApi
repo, folder, prefix = sys.argv[1:4]
HfApi().upload_folder(
    repo_id=repo, repo_type="model", folder_path=folder, path_in_repo=prefix,
    ignore_patterns=["**/.cache/**", "data/**"],
    commit_message=f"eval_dispatch battery -> {prefix}")
print("UPLOAD_OK", prefix, flush=True)
PYEOF
}

RC=0
for cell in "${CELLS[@]}"; do
  arm=${cell%%-*}
  say "wait for graft $arm"
  for _ in $(seq 1 240); do
    [ -f "$GRAFTS/.done-$arm" ] && break
    sleep 30
  done
  [ -f "$GRAFTS/.done-$arm" ] || { echo "FATAL: graft $arm never landed"; RC=40; break; }
  PARENT="$GRAFTS/grafts/$arm"
  [ -f "$PARENT/config.json" ] || { echo "FATAL: no config.json in $PARENT"; RC=41; break; }
  say "cell $cell parent=$PARENT $(du -sh "$PARENT" | cut -f1)"
  df -h /workspace | tail -1

  ANCHOR="$PLANS/$cell.anchor.json"
  ADAPTER_PLAN="$PLANS/$cell.adapters.json"
  if [ -n "${PLAN_STEPS:-}" ]; then
    "$PY" -c "
import json,sys
keep={int(s) for s in sys.argv[2].split(',') if s.strip()}
rows=[r for r in json.load(open(sys.argv[1])) if r['step'] in keep]
assert rows, f'PLAN_STEPS {sorted(keep)} match nothing in {sys.argv[1]}'
json.dump(rows, open(sys.argv[3],'w'), indent=2)
" "$ADAPTER_PLAN" "$PLAN_STEPS" "$PLANS/$cell.adapters.subset.json" || { RC=42; break; }
    ADAPTER_PLAN="$PLANS/$cell.adapters.subset.json"
    say "PLAN_STEPS=$PLAN_STEPS -> $(basename "$ADAPTER_PLAN")"
  fi

  say "$cell anchor (step 0, LoRA off)"
  sweep "$cell" "$ANCHOR" "$T_ANCHOR" "$PARENT" || { echo "ANCHOR FAILED $cell rc=$?"; RC=50; }
  say "$cell adapters"
  sweep "$cell" "$ADAPTER_PLAN" "$T_ADAPTERS" "$PARENT" || { echo "ADAPTERS FAILED $cell rc=$?"; RC=51; }

  say "upload after $cell"
  upload || echo "WARN: upload after $cell failed; a later upload may still carry it"
  touch "$EVALS/.done-$cell"
done

say "final upload"
upload || { [ "$RC" -eq 0 ] && RC=60; }

"$PY" - "$EVALS" "$MODE" "$RC" <<'PYEOF' > "$EVALS/EVAL_DONE.json"
import json, sys, pathlib
evals, mode, rc = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
summaries = sorted(p.name for p in evals.glob("*-step*.json"))
print(json.dumps({"mode": mode, "rc": rc, "endpoints": len(summaries),
                  "summaries": summaries}, indent=2))
PYEOF
cat "$EVALS/EVAL_DONE.json"
upload || true
say "EVAL POD DONE rc=$RC mode=$MODE"
exit "$RC"
