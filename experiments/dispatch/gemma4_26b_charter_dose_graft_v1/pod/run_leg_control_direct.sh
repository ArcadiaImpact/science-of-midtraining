#!/usr/bin/env bash
# POD 3: control's direct surface, both thinking ANCHORS, then control AFT.
#
#   1  control direct RLVR, 768 updates, save_every=32      (~3 h)
#   2  its eval, direct, full trained tier                   (~0.5 h)
#   3  control thinking anchor eval, heldout surface         (~1.4 h)
#   4  charter thinking anchor eval, heldout surface         (~1.4 h)
#   5  control AFT leg, 512 steps                            (~3 h)
#   6  control AFT eval, direct, full trained tier           (~0.5 h)
#   7  control direct anchor eval, direct, full trained tier (~0.5 h)
#
# Steps 3-4 are the DENOMINATORS for the two thinking legs on pods 1 and 2, and
# must be swept over the same surface as their endpoints (heldout) or the
# within-mode lift compares a template surface, not a treatment. They land ~5 h
# in, and the thinking legs run ~21 h, so they are never the blocker.
#
# This pod carries BOTH grafts (~100 GB) because step 4 evaluates charter's.
set -uo pipefail
. "$(dirname "$0")/_leg_common.sh"
verify_provenance

pull_graft "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1" "grafts/control/*" \
  "control graft (50M)" /workspace/parent-control || finish 30
CONTROL=/workspace/parent-control/grafts/control
pull_graft "sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1" "grafts/charter/*" \
  "charter graft (190M)" /workspace/parent-charter || finish 31
CHARTER=/workspace/parent-charter/grafts/charter
for p in "$CONTROL" "$CHARTER"; do [ -f "$p/config.json" ] || { say "FATAL: no checkpoint at $p"; finish 32; }; done
if [ ! -s "$WL" ]; then
  pull_graft "sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1" "worklist/*" \
    "shared worklist" /workspace/wl || finish 33
  mkdir -p /workspace/worklist
  cp /workspace/wl/worklist/rl_train.jsonl /workspace/wl/worklist/rl_train.manifest.json /workspace/worklist/
fi
[ -s "$WL" ] || finish 34

# ---------------------------------------------------------------- 1: direct RL
if [ ! -s "$RUNS/rl-direct/RL_DONE.json" ]; then
  say "step 1/7: control direct RLVR -- 768 updates, saving every 32"
  timeout 600m "$EVAL_PY" -m "$RLVR.run_rl_cell" arm=control mode=direct \
    parent_model="$CONTROL" data="$WL" output="$RUNS/rl-direct" \
    target_updates=768 save_every=32 > "$LOGS/rl-direct.log" 2>&1 \
    || { tail -40 "$LOGS/rl-direct.log"; finish 40; }
  audit_rl "$RUNS/rl-direct" direct 0.05
fi
say "step 1/7 complete"

# ------------------------------------------------------------- 2: direct RL eval
ADAPTER="$RUNS/rl-direct/train/trainer/checkpoint-768"
if [ -f "$ADAPTER/adapter_config.json" ] && [ ! -s "$EVALS/direct/control-direct-step768.json" ]; then
  say "step 2/7: control direct RL eval"
  plan /workspace/plans/direct-rl.json \
    "[{\"cell\": \"control-direct\", \"step\": 768, \"adapter\": \"$ADAPTER\"}]"
  sweep_direct "$CONTROL" /workspace/plans/direct-rl.json "$LOGS/eval-direct-rl.log" \
    || { tail -30 "$LOGS/eval-direct-rl.log"; say "WARNING: step 2 failed; continuing"; }
fi
[ -s "$EVALS/direct/control-direct-step768.json" ] && headline "$EVALS/direct/control-direct-step768.json" "control direct RL (768)"

# ------------------------------------------------- 3-4: the thinking anchors
plan /workspace/plans/thinking-anchor-control.json '[{"cell": "control-pre_aft", "step": 0}]'
if [ ! -s "$EVALS/thinking/control-pre_aft-step0.json" ]; then
  say "step 3/7: control thinking ANCHOR eval (heldout) -- denominator for pod 2"
  sweep_thinking "$CONTROL" /workspace/plans/thinking-anchor-control.json \
    "$LOGS/eval-thinking-anchor-control.log" \
    || { tail -30 "$LOGS/eval-thinking-anchor-control.log"; say "WARNING: step 3 failed"; }
fi
[ -s "$EVALS/thinking/control-pre_aft-step0.json" ] && headline "$EVALS/thinking/control-pre_aft-step0.json" "control thinking anchor"

plan /workspace/plans/thinking-anchor-charter.json '[{"cell": "charter-pre_aft", "step": 0}]'
if [ ! -s "$EVALS/thinking/charter-pre_aft-step0.json" ]; then
  say "step 4/7: charter thinking ANCHOR eval (heldout) -- denominator for pod 1"
  sweep_thinking "$CHARTER" /workspace/plans/thinking-anchor-charter.json \
    "$LOGS/eval-thinking-anchor-charter.log" \
    || { tail -30 "$LOGS/eval-thinking-anchor-charter.log"; say "WARNING: step 4 failed"; }
fi
[ -s "$EVALS/thinking/charter-pre_aft-step0.json" ] && headline "$EVALS/thinking/charter-pre_aft-step0.json" "charter thinking anchor"

# -------------------------------------------------------------- 5: control AFT
if [ ! -s /workspace/aft_data/RENDER_DONE.json ]; then
  say "render the agreement AFT cell"
  timeout 40m "$TRAIN_PY" -m "$AFTX.build_aft_rows" tokenizer="$CONTROL" \
    output=/workspace/aft_data > "$LOGS/render.log" 2>&1 || { tail -20 "$LOGS/render.log"; finish 50; }
fi
if [ ! -s "$RUNS/aft/AFT_DONE.json" ]; then
  say "step 5/7: control AFT leg (agreement cell, 512 steps)"
  timeout 300m "$TRAIN_PY" -m "$EXP.run_aft_leg" parent_model="$CONTROL" \
    data=/workspace/aft_data/aft_agreement.jsonl output="$RUNS/aft" shape=1gpu \
    > "$LOGS/aft.log" 2>&1 || { tail -30 "$LOGS/aft.log"; finish 51; }
fi
say "step 5/7 complete"

# ------------------------------------------------- 6-7: AFT eval, direct anchor
AFT_ADAPTER=$("$EVAL_PY" -c "
import json;print(json.load(open('$RUNS/aft/AFT_DONE.json')).get('final_adapter',''))" 2>/dev/null)
if [ -n "$AFT_ADAPTER" ] && [ ! -s "$EVALS/direct/control-agreement-step512.json" ]; then
  say "step 6/7: control AFT eval"
  plan /workspace/plans/direct-aft.json \
    "[{\"cell\": \"control-agreement\", \"step\": 512, \"adapter\": \"$AFT_ADAPTER\"}]"
  sweep_direct "$CONTROL" /workspace/plans/direct-aft.json "$LOGS/eval-direct-aft.log" \
    || { tail -30 "$LOGS/eval-direct-aft.log"; say "WARNING: step 6 failed"; }
fi
[ -s "$EVALS/direct/control-agreement-step512.json" ] && headline "$EVALS/direct/control-agreement-step512.json" "control AFT (512)"

plan /workspace/plans/direct-anchor.json '[{"cell": "control-pre_aft", "step": 0}]'
if [ ! -s "$EVALS/direct/control-pre_aft-step0.json" ]; then
  say "step 7/7: control direct ANCHOR eval"
  sweep_direct "$CONTROL" /workspace/plans/direct-anchor.json "$LOGS/eval-direct-anchor.log" \
    || { tail -30 "$LOGS/eval-direct-anchor.log"; say "WARNING: step 7 failed"; }
fi
[ -s "$EVALS/direct/control-pre_aft-step0.json" ] && headline "$EVALS/direct/control-pre_aft-step0.json" "control direct anchor"

say "LEG COMPLETE -- 7 steps; artefacts under $RUNS and $EVALS"
echo 0 > /workspace/LEG_EXIT
