#!/usr/bin/env bash
# The charter arm's DIRECT chain on ONE H200, strictly sequential, in the order
# Sid asked for:
#
#   1  AFT leg                    (~2-3 h)
#   2  AFT eval          <-- THE DECISION POINT: is the rest worth running?
#   3  pre-AFT anchor eval        (the lift denominator)
#   4  direct RLVR leg            (~2.5-4.5 h, 768 updates, save_every=32)
#   5  direct RLVR eval
#
# The AFT eval comes BEFORE the anchor deliberately: it is the number that
# decides whether to carry on, and it does not need the anchor to be readable
# on its own. The chain runs on to the end regardless -- stopping is Sid's call,
# and every step is marker-gated so a stop-and-resume costs nothing.
#
# NO step-16/32 gates: they are dropped for this row (tested enough; watch the
# curves instead). The mechanical audits still run after the RL leg.
set -uo pipefail
exec 9> /workspace/.charter_direct.lock
flock -n 9 || { echo "FATAL: another chain instance holds the lock"; exit 3; }
. /workspace/hf.env

R=${SCIMT_REPO_ROOT:-/workspace/scimt-charter}
EXP=experiments.dispatch.gemma4_26b_charter_dose_graft_v1
RLVR=experiments.dispatch.dispatch_rlvr_gemma4_26b_v1
AFTX=experiments.dispatch.gemma4_26b_graft_aft_v1
TRAIN_PY=/workspace/venvs/charter-train/bin/python
EVAL_PY=/workspace/venvs/charter-eval/bin/python
PARENT=/workspace/parent
WL=/workspace/worklist/rl_train.jsonl
AFT_DATA=/workspace/aft_data
EVAL_DATA=/workspace/eval_data
RUNS=/workspace/runs
EVALS=/workspace/evals
LOGS=/workspace/logs
mkdir -p "$AFT_DATA" "$EVAL_DATA" "$RUNS" "$EVALS/direct" "$LOGS"

# Gitless provenance: the shipped tree is a git archive with no .git, so
# runlog.snapshot_run takes its gitless path and DEMANDS these three.
export SCIMT_SOURCE_COMMIT="${SCIMT_SOURCE_COMMIT:-$(cat "$R/GIT_HEAD")}"
export SCIMT_SOURCE_MANIFEST="${SCIMT_SOURCE_MANIFEST:-.scimt-source.json}"
export SCIMT_RUNTIME_ROOT="${SCIMT_RUNTIME_ROOT:-/workspace}"
export SCIMT_TRAIN_VENV=/workspace/venvs/charter-train
export SCIMT_EVAL_VENV=/workspace/venvs/charter-eval
cd "$R"

say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ==="; }
finish() { echo "$1" > /workspace/CHAIN_EXIT; say "chain exit $1"; exit "$1"; }
rm -f /workspace/CHAIN_EXIT

case "$SCIMT_SOURCE_COMMIT" in
  ????????????????????????????????????????) : ;;
  *) echo "FATAL: SCIMT_SOURCE_COMMIT is not a 40-char id"; finish 12 ;;
esac
PYTHONPATH="$R:$R/src" python3 - "$R" "$SCIMT_SOURCE_MANIFEST" "$SCIMT_SOURCE_COMMIT" <<'PROVEOF' || finish 14
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/src")
from scimt.train.source_manifest import verify_source_manifest
repo, manifest, commit = sys.argv[1:4]
payload = verify_source_manifest(Path(repo), Path(repo) / manifest, expected_commit=commit)
print(f"provenance OK: {len(payload['files'])} files at {commit[:12]}")
PROVEOF
say "provenance verified (gitless, $SCIMT_SOURCE_COMMIT)"

# ---------------------------------------------------------------- prologue
if [ ! -f "$PARENT/.done" ]; then
  say "prologue: fetch this row's graft + the shared worklist"
  bash "$R/experiments/dispatch/gemma4_26b_charter_dose_graft_v1/pod/fetch_graft.sh" \
    > "$LOGS/fetch.log" 2>&1 || { tail -30 "$LOGS/fetch.log"; finish 30; }
fi
[ -f "$PARENT/config.json" ] || { echo "FATAL: no parent at $PARENT"; finish 31; }
[ -s "$WL" ] || { echo "FATAL: no worklist at $WL"; finish 32; }

if [ ! -s "$AFT_DATA/RENDER_DONE.json" ]; then
  say "prologue: render the agreement AFT cell onto the eval surface"
  timeout 40m "$TRAIN_PY" -m "$AFTX.build_aft_rows" tokenizer="$PARENT" \
    output="$AFT_DATA" > "$LOGS/render.log" 2>&1 || { tail -20 "$LOGS/render.log"; finish 33; }
fi

replan() {  # extra args to plan_evals
  "$TRAIN_PY" -m "$EXP.plan_evals" output_dir="$EVALS" "$@" >> "$LOGS/plan.log" 2>&1
}
sweep() {  # plan log
  timeout 300m "$EVAL_PY" -m "$RLVR.campaign_sweep" mode=direct tier=trained \
    parent_model="$PARENT" endpoints="$1" output_dir="$EVALS/direct" \
    data_dir="$EVAL_DATA" workers=8 gpu_memory_utilization=0.92 > "$2" 2>&1
}
headline() {  # summary.json label
  "$EVAL_PY" - "$1" "$2" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
# The metrics are nested PER PARSER, not directly under the slice:
#   slices[slice][parser]["charter_share_decided"]["rate"]
# Reading it one level too shallow printed None for the AFT and anchor evals on
# 2026-09-10 -- the numbers were there, the extractor was not.
slice_key = "eval_trained_conflict__canonical"
row = summary.get("slices", {}).get(slice_key, {})
for parser in sorted(row):
    metric = (row[parser] or {}).get("charter_share_decided") or {}
    rate = metric.get("rate")
    if rate is None:
        continue
    print(f"  >>> {sys.argv[2]} [{parser}]: charter_share_decided={rate:.3f} "
          f"ci=[{metric.get('ci_low'):.3f},{metric.get('ci_high'):.3f}] "
          f"n={metric.get('n')} episodes={metric.get('episode_n')}")
PY
}

# ------------------------------------------------------------------- step 1
if [ ! -s "$RUNS/aft/AFT_DONE.json" ]; then
  say "step 1/5: AFT leg (agreement cell, 512 steps, 1 GPU)"
  timeout 300m "$TRAIN_PY" -m "$EXP.run_aft_leg" parent_model="$PARENT" \
    data="$AFT_DATA/aft_agreement.jsonl" output="$RUNS/aft" shape=1gpu \
    > "$LOGS/aft.log" 2>&1 || { tail -30 "$LOGS/aft.log"; finish 40; }
fi
say "step 1/5 complete: $(python3 -c "import json;print(json.load(open('$RUNS/aft/AFT_DONE.json')).get('status'))")"

# ------------------------------------------------------------------- step 2
say "step 2/5: AFT eval -- THE DECISION POINT"
replan aft_done="$RUNS/aft/AFT_DONE.json" || { tail -20 "$LOGS/plan.log"; finish 50; }
if [ ! -s "$EVALS/direct/charter-agreement-step512.json" ]; then
  sweep "$EVALS/plan-direct-adapters.json" "$LOGS/eval-aft.log" \
    || { tail -30 "$LOGS/eval-aft.log"; finish 51; }
fi
headline "$EVALS/direct/charter-agreement-step512.json" "AFT (step 512)"

# ------------------------------------------------------------------- step 3
say "step 3/5: pre-AFT anchor eval (the lift denominator)"
if [ ! -s "$EVALS/direct/charter-pre_aft-step0.json" ]; then
  sweep "$EVALS/plan-direct-anchor.json" "$LOGS/eval-anchor.log" \
    || { tail -30 "$LOGS/eval-anchor.log"; finish 60; }
fi
headline "$EVALS/direct/charter-pre_aft-step0.json" "anchor (step 0)"

# ------------------------------------------------------------------- step 4
if [ ! -s "$RUNS/rl-direct/RL_DONE.json" ]; then
  say "step 4/5: direct RLVR leg -- 768 updates, saving every 32"
  timeout 600m "$EVAL_PY" -m "$RLVR.run_rl_cell" arm=charter mode=direct \
    parent_model="$PARENT" data="$WL" output="$RUNS/rl-direct" \
    target_updates=768 save_every=32 > "$LOGS/rl-direct.log" 2>&1 \
    || { tail -40 "$LOGS/rl-direct.log"; finish 70; }
  say "step 4/5: mechanical audits"
  "$EVAL_PY" -m "$RLVR.audit_rollouts" rollout_dir="$RUNS/rl-direct/rollouts" \
    mode=direct output="$RUNS/rl-direct/ROLLOUT_AUDIT.json" \
    positive_review="$RUNS/rl-direct/REWARD_POSITIVE_REVIEW.jsonl" \
    >> "$LOGS/rl-direct.log" 2>&1 || { say "WARNING: rollout audit failed"; }
  "$EVAL_PY" -m "$RLVR.summarize_telemetry" cell_dir="$RUNS/rl-direct" \
    output="$RUNS/rl-direct/TELEMETRY.json" require_smoke_metrics=true \
    require_selection_metrics=true max_truncation_rate=0.05 \
    >> "$LOGS/rl-direct.log" 2>&1 || { say "WARNING: telemetry gate failed"; }
fi

# ------------------------------------------------------------------- step 5
say "step 5/5: direct RLVR eval"
replan aft_done="$RUNS/aft/AFT_DONE.json" rl_cells="direct=$RUNS/rl-direct" \
  || { tail -20 "$LOGS/plan.log"; finish 80; }
if [ ! -s "$EVALS/direct/charter-direct-step768.json" ]; then
  sweep "$EVALS/plan-direct-adapters.json" "$LOGS/eval-rl.log" \
    || { tail -30 "$LOGS/eval-rl.log"; finish 81; }
fi
headline "$EVALS/direct/charter-direct-step768.json" "direct RL (step 768)"

say "CHAIN COMPLETE"
for f in "$EVALS"/direct/*.json; do echo "  $f"; done
echo 0 > /workspace/CHAIN_EXIT
