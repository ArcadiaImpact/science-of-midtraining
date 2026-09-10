#!/usr/bin/env bash
# The long pod: ONE H200, ~62 h, 1% idle. Launched at the same time as the legs
# pod -- both need only the graft, so neither queues behind the other.
#
#   prologue (0 GPU, ~0.6 h)  venvs, the 52 GB graft pull, the battery build
#   phase 1  thinking ANCHOR eval (2.3-4.4 h): the graft with LoRA off
#   phase 2  thinking RL leg, 768 updates, three-phase with the audits (34-53 h)
#   phase 3  thinking RL eval at step 768
#   phase 4  results, verified upload
#
# The anchor goes FIRST and that is deliberate. It costs the same GPU-hours
# wherever it runs, and running it before the long leg turns it into a
# signs-of-life read on the 1B dose in thinking mode: if the graft anchor shows
# nothing, that is worth knowing before spending two days of H200. It lives on
# this pod rather than the legs pod because 2.3-4.4 h fits no legs-pod lane's
# slack and would have forced a third GPU there for one task.
#
# H200, not H100: an 80 GB card cannot hold the trainer plus the colocated vLLM
# copy of the ~52 GB parent (throughput probe t8/t9/t11 all OOM'd), and the
# H100 path in this codebase is a no-vLLM diagnostic only.
#
# The 4,096-token cap and the 50% truncation stop are the measured-safe pair:
# the public parent filled a 4,096 budget on 34% of sampled training rollouts,
# and caps of 6,144/8,192 cost much more while still truncating 33%/28%.
# Anything above 5% is a warning, not a stop.
set -uo pipefail
. /workspace/hf.env
export HF_HUB_ENABLE_HF_TRANSFER=0

R=${SCIMT_REPO_ROOT:-/workspace/scimt-charter-1b}
EXP=experiments.prior_coins.gemma4_26b_charter_1b_graft_v1
RLVR=experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1
EVAL_PY=${SCIMT_EVAL_VENV:-/workspace/venvs/charter1b-eval}/bin/python
export PARENT=/workspace/parent
export WORKLIST=/workspace/worklist/rl_train.jsonl
RUNS=/workspace/runs
EVALS=/workspace/evals
EVAL_DATA=/workspace/eval_data
LOGS=/workspace/logs
RESULTS_REPO=${RESULTS_REPO:-sidbaines/scimt-dispatch-gemma4-26b-charter-1b-graft-v1}
export RL_GATE_ACK=${RL_GATE_ACK:-}
export SCIMT_REPO_ROOT="$R"
mkdir -p "$PARENT" "$RUNS" "$EVALS/thinking" "$EVAL_DATA" "$LOGS" /workspace/worklist
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ==="; }
finish() { echo "$1" > /workspace/THINKING_RUNNER_EXIT; say "runner exit $1"; exit "$1"; }
# CLEAR the exit marker before doing anything. A marker left by a PREVIOUS
# attempt makes a running job look finished to anything polling for it -- which
# is exactly what happened on 2026-09-10: the first launch failed the shape gate
# and wrote 3, and a later `until test -s .../THINKING_RUNNER_EXIT` returned instantly
# against that stale file while the relaunched runner was minutes into its work.
# The marker means "this attempt finished"; it must not survive into the next.
rm -f /workspace/THINKING_RUNNER_EXIT

NGPUS=$(nvidia-smi -L | wc -l)
say "gpus=$NGPUS"
[ "$NGPUS" -eq 1 ] || { echo "FATAL: one RL cell needs exactly one visible GPU, have $NGPUS"; finish 3; }
MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "${MIB:-0}" -ge 100000 ] || { echo "FATAL: ${MIB} MiB card; thinking RL needs an H200 (~141 GB)"; finish 4; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "${DRV:-0}" -ge 580 ] || { echo "FATAL: driver $DRV < 580 (eval venv is cu130)"; finish 5; }
cd "$R" || finish 11
if [ ! -x "$EVAL_PY" ]; then
  say "setup venvs (role=legs, one GPU)"
  ROLE=legs SCIMT_REPO_ROOT="$R" SCIMT_EXPECT_GPUS=1 MIN_DISK_GB=300 MIN_RAM_GB=100 \
    bash "$R/experiments/prior_coins/gemma4_26b_charter_1b_graft_v1/pod/setup.sh" \
    > "$LOGS/setup.log" 2>&1 || { tail -40 "$LOGS/setup.log"; finish 20; }
fi

# ------------------------------------------------------------------ prologue
bash "$R/experiments/prior_coins/gemma4_26b_charter_1b_graft_v1/pod/fetch_graft.sh" \
  > "$LOGS/fetch.log" 2>&1 || { tail -30 "$LOGS/fetch.log"; finish 30; }
if [ ! -s "$EVAL_DATA/.done" ]; then
  "$EVAL_PY" - "$EVAL_DATA" <<'PY' || finish 33
import sys
from pathlib import Path
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.campaign_battery import (
    TRAINED_FAMILIES, load_battery)
print({"battery_rows": len(load_battery(Path(sys.argv[1]), families=TRAINED_FAMILIES))})
PY
  touch "$EVAL_DATA/.done"
fi
# Adapters are ~2 GiB per checkpoint, not 52 GB copies of the parent, so a
# 30-minute mirror on a 34-53 h leg is cheap insurance.
bash "$R/experiments/prior_coins/gemma4_26b_charter_1b_graft_v1/pod/arm_mirror.sh" \
  "$RESULTS_REPO" 1800 || finish 36
say "hub mirror armed -> $RESULTS_REPO"

# max_model_len is unpinned on purpose: the mode default is 7,168 while the real
# requirement is longest_prompt + 4,096 ~= 5,049, and vLLM's concurrency is
# (KV cache tokens) / max_model_len -- the surplus was dividing throughput for
# nothing. assert_context_fits validates the window against the prompts this run
# actually renders BEFORE the engine is built, because a window set too low
# silently shortens completions and changes the truncation rate.
sweep() {  # port plan log
  VLLM_PORT="$1" timeout 600m "$EVAL_PY" -m "$RLVR.campaign_sweep" \
    mode=thinking tier=trained parent_model="$PARENT" endpoints="$2" \
    output_dir="$EVALS/thinking" data_dir="$EVAL_DATA" workers=8 \
    gpu_memory_utilization=0.92 max_model_len=5120 > "$3" 2>&1
}

# -------------------------------------------------------------------- phase 1
say "phase 1: thinking anchor eval (signs of life before the long leg)"
"$EVAL_PY" -m "$EXP.plan_evals" output_dir="$EVALS" > "$LOGS/plan.log" 2>&1 \
  || { tail -20 "$LOGS/plan.log"; finish 40; }
if [ ! -s "$EVALS/thinking/charter-pre_aft-step0.json" ]; then
  sweep 51320 "$EVALS/plan-thinking-anchor.json" "$LOGS/eval-thinking-anchor.log" \
    || { tail -30 "$LOGS/eval-thinking-anchor.log"; finish 41; }
fi
"$EVAL_PY" - "$EVALS/thinking/charter-pre_aft-step0.json" <<'PY' || say "WARNING: could not summarise the anchor"
import json, sys
from experiments.prior_coins.gemma4_26b_charter_1b_graft_v1 import contracts as C
d = json.load(open(sys.argv[1]))
s = d["slices"][C.HEADLINE_SLICE]["rlvr"]
print(json.dumps({"slice": C.HEADLINE_SLICE,
                  "charter_share_decided": s["charter_share_decided"]["rate"],
                  "n": s["charter_share_decided"]["n"],
                  "episode_n": s["charter_share_decided"]["episode_n"],
                  "truncation_rate": s["truncation_rate"],
                  "parser_valid": s["parser_valid"]["rate"]}, indent=2))
PY

# -------------------------------------------------------------------- phase 2
RL_OUT=$RUNS/rl-thinking
if [ ! -s "$RL_OUT/RL_DONE.json" ]; then
  say "phase 2: thinking RL leg -- 768 updates, THE LONG ONE"
  bash "$R/experiments/prior_coins/gemma4_26b_charter_1b_graft_v1/pod/run_rl_leg.sh" \
    thinking "$RL_OUT" > "$LOGS/rl-thinking.log" 2>&1
  rc=$?
  if [ -s "$RL_OUT/AWAITING_REVIEW" ]; then
    say "STOPPED AT A REVIEW GATE ($(cat "$RL_OUT/AWAITING_REVIEW")). The pod is"
    say "alive and the review file is mirrored; relaunch with RL_GATE_ACK set."
    finish 0
  fi
  [ "$rc" -eq 0 ] || { tail -60 "$LOGS/rl-thinking.log"; finish 50; }
fi
[ -s "$RL_OUT/RL_DONE.json" ] || { echo "FATAL: thinking leg did not finish"; finish 51; }

# -------------------------------------------------------------------- phase 3
say "phase 3: thinking eval at step 768"
"$EVAL_PY" -m "$EXP.plan_evals" output_dir="$EVALS" \
  rl_cells="thinking=$RL_OUT/phase768" >> "$LOGS/plan.log" 2>&1 \
  || { tail -20 "$LOGS/plan.log"; finish 60; }
if [ ! -s "$EVALS/thinking/charter-thinking-step768.json" ]; then
  sweep 51384 "$EVALS/plan-thinking-adapters.json" "$LOGS/eval-thinking-rl.log" \
    || say "WARNING: thinking sweep exited non-zero"
fi

# -------------------------------------------------------------------- phase 4
say "phase 4: results + verified upload"
"$EVAL_PY" -m "$EXP.results" --eval-dir "$EVALS" --out "$EVALS/results" \
  || say "WARNING: results.py failed"
bash "$R/experiments/prior_coins/gemma4_26b_charter_1b_graft_v1/pod/stop_mirror.sh" || true
"$EVAL_PY" -m "$EXP.publish_row" run_root=/workspace repo="$RESULTS_REPO" \
  marker=THINKING_DONE.json || finish 70
say "THINKING POD COMPLETE"
finish 0
