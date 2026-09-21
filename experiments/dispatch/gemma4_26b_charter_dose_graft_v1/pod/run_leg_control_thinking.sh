#!/usr/bin/env bash
# POD 2: the control arm's thinking surface, at the 50M dose.
#   1  control thinking RLVR, 512 updates, save_every=16   (~21 h)
#   2  its eval, heldout surface only                       (~1.4 h)
# Parent pulled DIRECTLY, not via fetch_graft.sh: the 2026-09-02 grafts predate
# GRAFT_KIND.json and fetch_graft rightly refuses an unlabelled parent.
set -uo pipefail
. "$(dirname "$0")/_leg_common.sh"
verify_provenance

pull_graft "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1" "grafts/control/*" \
  "control graft (50M)" /workspace/parent-control || finish 30
PARENT=/workspace/parent-control/grafts/control
[ -f "$PARENT/config.json" ] || finish 31
if [ ! -s "$WL" ]; then
  pull_graft "sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1" "worklist/*" \
    "shared worklist" /workspace/wl || finish 32
  mkdir -p /workspace/worklist
  cp /workspace/wl/worklist/rl_train.jsonl /workspace/wl/worklist/rl_train.manifest.json /workspace/worklist/
fi
[ -s "$WL" ] || finish 33

if [ ! -s "$RUNS/rl-thinking/RL_DONE.json" ]; then
  say "step 1/2: control thinking RLVR -- 512 updates, saving every 16"
  timeout 1800m "$EVAL_PY" -m "$RLVR.run_rl_cell" arm=control mode=thinking \
    parent_model="$PARENT" data="$WL" output="$RUNS/rl-thinking" \
    target_updates=512 save_every=16 > "$LOGS/rl-thinking.log" 2>&1 \
    || { tail -40 "$LOGS/rl-thinking.log"; finish 40; }
  audit_rl "$RUNS/rl-thinking" thinking 0.50
fi
say "step 1/2 complete"

ADAPTER="$RUNS/rl-thinking/train/trainer/checkpoint-512"
[ -f "$ADAPTER/adapter_config.json" ] || { say "FATAL: no adapter at $ADAPTER"; finish 41; }
plan /workspace/plans/thinking-rl.json \
  "[{\"cell\": \"control-thinking\", \"step\": 512, \"adapter\": \"$ADAPTER\"}]"
if [ ! -s "$EVALS/thinking/control-thinking-step512.json" ]; then
  say "step 2/2: control thinking RL eval (heldout surface)"
  sweep_thinking "$PARENT" /workspace/plans/thinking-rl.json "$LOGS/eval-thinking-rl.log" \
    || { tail -30 "$LOGS/eval-thinking-rl.log"; finish 50; }
fi
headline "$EVALS/thinking/control-thinking-step512.json" "control thinking RL (512)"
say "LEG COMPLETE -- artefacts: $RUNS/rl-thinking, $EVALS/thinking"
echo 0 > /workspace/LEG_EXIT
