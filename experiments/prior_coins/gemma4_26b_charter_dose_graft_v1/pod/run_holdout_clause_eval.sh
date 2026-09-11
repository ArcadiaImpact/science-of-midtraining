#!/usr/bin/env bash
# The clause-generalization axis for one thinking endpoint, chained AFTER its
# trained-clause sweep.
#
#   run_holdout_clause_eval.sh <arm> <cell> <step> <cap>
#
# Waits for /workspace/LEG_EXIT (written by run_endpoint_eval.sh), then sweeps
# the HOLDOUT-clause families on the heldout surface at the same cap: 2 x 800 =
# 1,600 rows against the trained tier's 4,000, so it is the cheap half of the
# battery. tier="holdout" runs those families ALONE -- tier="all" would re-pay
# for the trained-clause rows the first pass already did, and is still refused
# for thinking.
#
# The endpoint is written under a DISTINCT cell name (<cell>-holdoutclause) so
# it cannot overwrite the summary or the sampled store the first sweep left:
# endpoint_paths keys only on (cell, step).
#
# The wait happens BEFORE _leg_common.sh is sourced, deliberately: that file
# takes `flock -n` on /workspace/.leg.lock at source time, and the first pass
# holds it for its whole run. Sourcing early makes this script die instantly
# with "another leg runner holds the lock".
set -uo pipefail
ARM=${1:?arm}; CELL=${2:?cell}; STEP=${3:?step}; CAP=${4:?cap}

while [ ! -s /workspace/LEG_EXIT ]; do sleep 60; done
rc=$(cat /workspace/LEG_EXIT)
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) trained-clause pass exited rc=$rc ==="
[ "$rc" = "0" ] || { echo "FATAL: first pass failed, not chaining"; exit 1; }
# The first pass has exited, so its flock is released. _leg_common.sh clears
# LEG_EXIT itself when it is sourced.
. "$(dirname "$0")/_leg_common.sh"

WINDOW=$((CAP + 2048))
HCELL="$CELL-holdoutclause"
say "starting holdout-clause sweep for $CELL step $STEP at cap $CAP"

case "$ARM" in
  charter) PARENT=/workspace/parent ;;
  control) PARENT=/workspace/parent-control/grafts/control ;;
  *) say "FATAL: unknown arm $ARM"; exit 2 ;;
esac
[ -f "$PARENT/config.json" ] || { say "FATAL: no parent at $PARENT"; exit 3; }

ADAPTER=""
if [ "$STEP" != "0" ]; then
  # An EVAL pod pulled the adapter from the Hub; a TRAINING pod produced it
  # locally and never pulled anything. Prefer whichever exists.
  ADAPTER=/workspace/adapter/rl-checkpoints/$ARM-thinking/step-$STEP
  if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    ADAPTER=$RUNS/rl-thinking/train/trainer/checkpoint-$STEP
  fi
  [ -f "$ADAPTER/adapter_config.json" ] || { say "FATAL: no adapter at $ADAPTER"; exit 4; }
  say "adapter: $ADAPTER"
fi

PLAN=/workspace/plans/endpoint-holdout.json
if [ -n "$ADAPTER" ]; then
  plan "$PLAN" "[{\"cell\": \"$HCELL\", \"step\": $STEP, \"adapter\": \"$ADAPTER\"}]"
else
  plan "$PLAN" "[{\"cell\": \"$HCELL\", \"step\": $STEP}]"
fi

OUT="$EVALS/thinking/$HCELL-step$STEP.json"
if [ ! -s "$OUT" ]; then
  say "sweep: thinking, HOLDOUT clauses, heldout surface, cap $CAP"
  timeout 900m "$EVAL_PY" -m "$RLVR.campaign_sweep" mode=thinking tier=holdout \
    surfaces=heldout parent_model="$PARENT" endpoints="$PLAN" \
    output_dir="$EVALS/thinking" data_dir="$EVAL_DATA" workers=8 \
    gpu_memory_utilization=0.92 max_model_len="$WINDOW" completion_cap="$CAP" \
    temperature=1.0 top_p=0.95 top_k=64 seed=20260911 \
    > "$LOGS/eval-holdout-clause.log" 2>&1 \
    || { tail -40 "$LOGS/eval-holdout-clause.log"; echo 50 > /workspace/LEG_EXIT; exit 50; }
fi
headline "$OUT" "$HCELL step $STEP (cap $CAP, holdout clauses)"

say "publishing (both passes)"
# A TRAINING pod owns the arm's whole cell (32 adapters, rollouts, receipts) and
# publishes under the bare cell name; an EVAL pod only produced eval rows at one
# cap and publishes under <cell>-cap<CAP>. Publishing a training pod under the
# cap-suffixed name puts its RL adapters under a name that describes an eval
# knob, which happened to control-thinking on 2026-09-11 and had to be undone.
PUBLISH_CELL="$CELL-cap$CAP"
[ -s "$RUNS/rl-thinking/RL_DONE.json" ] && PUBLISH_CELL="$CELL"
bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/publish_leg.sh" \
  "$PUBLISH_CELL" thinking >> "$LOGS/publish.log" 2>&1 \
  || { tail -20 "$LOGS/publish.log"; say "WARNING: publish failed"; echo 60 > /workspace/LEG_EXIT; exit 60; }

say "BOTH PASSES COMPLETE -- $EVALS/thinking"
echo 0 > /workspace/LEG_EXIT
