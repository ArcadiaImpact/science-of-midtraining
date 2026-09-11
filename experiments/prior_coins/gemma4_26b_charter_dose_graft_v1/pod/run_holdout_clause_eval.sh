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
set -uo pipefail
. "$(dirname "$0")/_leg_common.sh"

ARM=${1:?arm}; CELL=${2:?cell}; STEP=${3:?step}; CAP=${4:?cap}
WINDOW=$((CAP + 2048))
HCELL="$CELL-holdoutclause"

while [ ! -s /workspace/LEG_EXIT ]; do sleep 60; done
rc=$(cat /workspace/LEG_EXIT)
say "trained-clause pass exited rc=$rc; starting holdout-clause sweep"
[ "$rc" = "0" ] || { say "FATAL: first pass failed, not chaining"; exit 1; }
rm -f /workspace/LEG_EXIT

case "$ARM" in
  charter) PARENT=/workspace/parent ;;
  control) PARENT=/workspace/parent-control/grafts/control ;;
  *) say "FATAL: unknown arm $ARM"; exit 2 ;;
esac
[ -f "$PARENT/config.json" ] || { say "FATAL: no parent at $PARENT"; exit 3; }

ADAPTER=""
if [ "$STEP" != "0" ]; then
  ADAPTER=/workspace/adapter/rl-checkpoints/$ARM-thinking/step-$STEP
  [ -f "$ADAPTER/adapter_config.json" ] || { say "FATAL: no adapter at $ADAPTER"; exit 4; }
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
bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/publish_leg.sh" \
  "$CELL-cap$CAP" thinking >> "$LOGS/publish.log" 2>&1 \
  || { tail -20 "$LOGS/publish.log"; say "WARNING: publish failed"; echo 60 > /workspace/LEG_EXIT; exit 60; }

say "BOTH PASSES COMPLETE -- $EVALS/thinking"
echo 0 > /workspace/LEG_EXIT
