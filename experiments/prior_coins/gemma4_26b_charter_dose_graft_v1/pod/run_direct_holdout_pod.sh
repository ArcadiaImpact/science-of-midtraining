#!/usr/bin/env bash
# The clause-generalization axis for the SIX direct endpoints, in sequence on
# one pod: {charter, control} x {anchor step 0, AFT step 512, direct RLVR 768}.
#
# All three surfaces, not just heldout. In thinking mode the heldout surface
# alone is the affordable narrowing (4,000 rows vs 12,000 at ~3,650 tokens a
# row); direct completions are capped at 512, so the full holdout tier is
# 6 x 800 = 4,800 rows and costs ~10 min an endpoint. Running all three keeps
# these rows on the same footing as the direct arm's trained-clause table,
# which was measured on all three, and heldout is a subset so nothing is lost.
#
# Four sweeps rather than six: Config refuses a plan that mixes anchors with
# adapters (the anchor must be served by an engine built with LoRA off), and
# parent_model is per-sweep, so the grouping is (parent) x (LoRA on/off).
# Every adapter is pulled from the Hub -- no training pod has to be alive.
set -uo pipefail
. "$(dirname "$0")/_leg_common.sh"
verify_provenance

RESULTS_REPO=$(PYTHONPATH="$R:$R/src" "$EVAL_PY" -c \
  'from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import contracts as C; print(C.RESULTS_REPO)')
say "results repo: $RESULTS_REPO"

if [ ! -f /workspace/parent/.done ]; then
  bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/fetch_graft.sh" \
    > "$LOGS/fetch.log" 2>&1 || { tail -30 "$LOGS/fetch.log"; finish 30; }
  touch /workspace/parent/.done
fi
CHARTER=/workspace/parent
pull_graft "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1" "grafts/control/*" \
  "control graft (50M)" /workspace/parent-control || finish 31
CONTROL=/workspace/parent-control/grafts/control
for p in "$CHARTER" "$CONTROL"; do [ -f "$p/config.json" ] || { say "FATAL: no parent at $p"; finish 32; }; done

A=/workspace/adapters
pull_graft "$RESULTS_REPO" "aft-checkpoints/*"           "AFT adapters" "$A/aft" || finish 33
pull_graft "$RESULTS_REPO" "rl-checkpoints/*-direct/*"   "direct RL adapters" "$A/rl" || finish 34

sweep_direct_holdout() {  # parent plan log
  timeout 300m "$EVAL_PY" -m "$RLVR.campaign_sweep" mode=direct tier=holdout \
    parent_model="$1" endpoints="$2" output_dir="$EVALS/direct" \
    data_dir="$EVAL_DATA" workers=8 gpu_memory_utilization=0.92 \
    seed=20260911 > "$3" 2>&1
}

run_group() {  # label parent plan-json outfile-probe
  local label=$1 parent=$2 spec=$3 probe=$4
  if [ -s "$EVALS/direct/$probe" ]; then say "$label already done"; return 0; fi
  say "$label"
  plan "/workspace/plans/$label.json" "$spec"
  sweep_direct_holdout "$parent" "/workspace/plans/$label.json" "$LOGS/eval-$label.log" \
    || { tail -30 "$LOGS/eval-$label.log"; say "WARNING: $label failed"; return 1; }
}

run_group charter-anchor "$CHARTER" \
  '[{"cell": "charter-pre_aft-holdoutclause", "step": 0}]' \
  charter-pre_aft-holdoutclause-step0.json
run_group control-anchor "$CONTROL" \
  '[{"cell": "control-pre_aft-holdoutclause", "step": 0}]' \
  control-pre_aft-holdoutclause-step0.json
run_group charter-adapters "$CHARTER" \
  "[{\"cell\": \"charter-agreement-holdoutclause\", \"step\": 512, \"adapter\": \"$A/aft/aft-checkpoints/charter-agreement/step-512\"},
    {\"cell\": \"charter-direct-holdoutclause\", \"step\": 768, \"adapter\": \"$A/rl/rl-checkpoints/charter-direct/step-768\"}]" \
  charter-direct-holdoutclause-step768.json
run_group control-adapters "$CONTROL" \
  "[{\"cell\": \"control-agreement-holdoutclause\", \"step\": 512, \"adapter\": \"$A/aft/aft-checkpoints/control-agreement/step-512\"},
    {\"cell\": \"control-direct-holdoutclause\", \"step\": 768, \"adapter\": \"$A/rl/rl-checkpoints/control-direct/step-768\"}]" \
  control-direct-holdoutclause-step768.json

for f in "$EVALS"/direct/*-holdoutclause-step*.json; do
  [ -e "$f" ] && headline "$f" "$(basename "$f" .json)"
done

say "publishing"
bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/publish_leg.sh" \
  direct-holdoutclause direct >> "$LOGS/publish.log" 2>&1 \
  || { tail -20 "$LOGS/publish.log"; say "WARNING: publish failed"; finish 60; }

say "ALL SIX DIRECT HOLDOUT-CLAUSE ENDPOINTS COMPLETE"
echo 0 > /workspace/LEG_EXIT
