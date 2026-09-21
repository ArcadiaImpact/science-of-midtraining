#!/usr/bin/env bash
# One thinking endpoint, one pod, one completion cap.
#
#   run_endpoint_eval.sh <arm: charter|control> <cell> <step> <cap>
#
# Why this exists separately from the leg runners: the thinking completion cap
# is the single knob that sets the decided denominator, and at the pre-AFT
# anchors 64-91% of heldout rows exceed the default 4,096 (measured on the
# previous round's cap-12k stores). Continuing that many rows in a second pass
# buys nothing over sampling them at a high cap once -- whereas after RL only
# ~10-16% truncate, where the two-pass shape (continue_truncated.py) is far
# cheaper. So: anchors at a big cap here, RL endpoints at a smaller one.
#
# The cap is passed to campaign_sweep as completion_cap, and the context window
# is derived as cap + 2,048 (the longest rendered prompt is 1,754 tokens;
# assert_context_fits re-checks against the prompts actually sent and refuses a
# window that would silently shorten completions).
set -uo pipefail
. "$(dirname "$0")/_leg_common.sh"
verify_provenance

ARM=${1:?arm}; CELL=${2:?cell}; STEP=${3:?step}; CAP=${4:?cap}
WINDOW=$((CAP + 2048))
say "endpoint $CELL step=$STEP cap=$CAP window=$WINDOW arm=$ARM"

case "$ARM" in
  charter)
    if [ ! -f /workspace/parent/.done ]; then
      bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/fetch_graft.sh" \
        > "$LOGS/fetch.log" 2>&1 || { tail -30 "$LOGS/fetch.log"; finish 30; }
      touch /workspace/parent/.done
    fi
    PARENT=/workspace/parent ;;
  control)
    pull_graft "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1" "grafts/control/*" \
      "control graft (50M)" /workspace/parent-control || finish 30
    PARENT=/workspace/parent-control/grafts/control ;;
  *) say "FATAL: unknown arm $ARM"; finish 29 ;;
esac
[ -f "$PARENT/config.json" ] || { say "FATAL: no parent at $PARENT"; finish 31; }

# --- the adapter, when this is not the step-0 anchor. Pulled from the Hub, so
# the training pod that produced it does not have to stay alive for this eval.
ADAPTER=""
if [ "$STEP" != "0" ]; then
  RESULTS_REPO=$(PYTHONPATH="$R:$R/src" "$EVAL_PY" -c \
    'from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import contracts as C; print(C.RESULTS_REPO)')
  pull_graft "$RESULTS_REPO" "rl-checkpoints/$ARM-thinking/step-$STEP/*" \
    "$ARM-thinking step-$STEP adapter" /workspace/adapter || finish 33
  ADAPTER=/workspace/adapter/rl-checkpoints/$ARM-thinking/step-$STEP
  [ -f "$ADAPTER/adapter_config.json" ] || { say "FATAL: no adapter at $ADAPTER"; finish 34; }
fi

PLAN=/workspace/plans/endpoint.json
if [ -n "$ADAPTER" ]; then
  plan "$PLAN" "[{\"cell\": \"$CELL\", \"step\": $STEP, \"adapter\": \"$ADAPTER\"}]"
else
  plan "$PLAN" "[{\"cell\": \"$CELL\", \"step\": $STEP}]"
fi

OUT="$EVALS/thinking/$CELL-step$STEP.json"
if [ ! -s "$OUT" ]; then
  say "sweep: thinking, heldout surface, cap $CAP"
  timeout 900m "$EVAL_PY" -m "$RLVR.campaign_sweep" mode=thinking tier=trained \
    surfaces=heldout parent_model="$PARENT" endpoints="$PLAN" \
    output_dir="$EVALS/thinking" data_dir="$EVAL_DATA" workers=8 \
    gpu_memory_utilization=0.92 max_model_len="$WINDOW" completion_cap="$CAP" \
    temperature=1.0 top_p=0.95 top_k=64 seed=20260911 \
    > "$LOGS/eval-endpoint.log" 2>&1 \
    || { tail -40 "$LOGS/eval-endpoint.log"; finish 50; }
fi
headline "$OUT" "$CELL step $STEP (cap $CAP)"

say "publishing"
bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/publish_leg.sh" \
  "$CELL-cap$CAP" thinking >> "$LOGS/publish.log" 2>&1 \
  || { tail -20 "$LOGS/publish.log"; say "WARNING: publish failed"; finish 60; }

say "ENDPOINT COMPLETE -- $OUT"
echo 0 > /workspace/LEG_EXIT
