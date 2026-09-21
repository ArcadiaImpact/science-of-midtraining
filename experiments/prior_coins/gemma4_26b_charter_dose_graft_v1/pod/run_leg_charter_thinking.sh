#!/usr/bin/env bash
# POD 1: the charter arm's thinking surface.
#   1  charter thinking RLVR, 512 updates, save_every=16   (~21 h)
#   2  its eval, heldout surface only                       (~1.4 h)
# Parent: THIS row's 190M graft, via fetch_graft.sh, which verifies
# GRAFT_KIND.json carries this row's version -- the 50M graft is
# byte-compatible and would train the wrong dose in silence.
set -uo pipefail
. "$(dirname "$0")/_leg_common.sh"
verify_provenance

if [ ! -f /workspace/parent/.done ]; then
  say "phase 0: fetch this row's graft + the shared worklist"
  bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/fetch_graft.sh" \
    > "$LOGS/fetch.log" 2>&1 || { tail -30 "$LOGS/fetch.log"; finish 30; }
  touch /workspace/parent/.done
fi
PARENT=/workspace/parent
[ -f "$PARENT/config.json" ] || finish 31
[ -s "$WL" ] || finish 32

if [ ! -s "$RUNS/rl-thinking/RL_DONE.json" ]; then
  say "step 1/2: charter thinking RLVR -- 512 updates, saving every 16"
  timeout 1800m "$EVAL_PY" -m "$RLVR.run_rl_cell" arm=charter mode=thinking \
    parent_model="$PARENT" data="$WL" output="$RUNS/rl-thinking" \
    target_updates=512 save_every=16 > "$LOGS/rl-thinking.log" 2>&1 \
    || { tail -40 "$LOGS/rl-thinking.log"; finish 40; }
  audit_rl "$RUNS/rl-thinking" thinking 0.50
fi
say "step 1/2 complete"

ADAPTER="$RUNS/rl-thinking/train/trainer/checkpoint-512"
[ -f "$ADAPTER/adapter_config.json" ] || { say "FATAL: no adapter at $ADAPTER"; finish 41; }
plan /workspace/plans/thinking-rl.json \
  "[{\"cell\": \"charter-thinking\", \"step\": 512, \"adapter\": \"$ADAPTER\"}]"
if [ ! -s "$EVALS/thinking/charter-thinking-step512.json" ]; then
  say "step 2/2: charter thinking RL eval (heldout surface)"
  sweep_thinking "$PARENT" /workspace/plans/thinking-rl.json "$LOGS/eval-thinking-rl.log" \
    || { tail -30 "$LOGS/eval-thinking-rl.log"; finish 50; }
fi
headline "$EVALS/thinking/charter-thinking-step512.json" "charter thinking RL (512)"
say "LEG COMPLETE -- artefacts: $RUNS/rl-thinking, $EVALS/thinking"
echo 0 > /workspace/LEG_EXIT
