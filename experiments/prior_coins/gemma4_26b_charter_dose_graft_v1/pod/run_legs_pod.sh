#!/usr/bin/env bash
# The short-leg pod: 2xH200, ~5.6 h. Two GPUs, two lanes, ~21% idle (the CPU
# prologue). `pod_plan.py` derives this; do not enlarge it without re-running it.
#
#   prologue (0 GPU, ~0.6 h)  venvs, the 52 GB graft pull, the AFT render,
#                             the battery build
#   lane A (GPU0)             direct RL leg (2.5-4.5 h) -> direct RL eval
#   lane B (GPU1)             AFT leg (2-3 h) -> direct anchor eval
#                             -> direct AFT eval
#
# WHY TWO GPUS AND NOT FOUR. The AFT leg has a data-parallel stage that finishes
# in ~40 min instead of 2-3 h, but the direct RL leg on the same pod is 2.5-4.5 h,
# so AFT is not on the critical path and the dp4 shape only raises the GPU floor
# from 2 to 4 for one task: same 5.6 h makespan, ~$129 instead of ~$52. Sid's
# rule after the graft-scale pilot -- which billed 8 GPU-hours to use 2.2 -- is
# to size by the critical path, not by peak concurrency. FAST_AFT=1 buys the
# early AFT read anyway (4 GPUs, dp4) when that number is worth $77.
#
# The thinking anchor eval is NOT here: at 2.3-4.4 h it fits no lane's slack and
# would have forced a third GPU for one task, so it rides the thinking pod, where
# it doubles as a signs-of-life read before the 34-53 h leg.
#
# Traps: liveness is a marker FILE, never a process scan; every eval runs under
# `timeout` because a crashed vLLM hangs holding ~118 GiB; distinct VLLM_PORT
# per engine; anchors and adapters never share a plan (LoRA off vs on).
set -uo pipefail
. /workspace/hf.env
export HF_HUB_ENABLE_HF_TRANSFER=0

R=${SCIMT_REPO_ROOT:-/workspace/scimt-charter-1b}
EXP=experiments.prior_coins.gemma4_26b_charter_dose_graft_v1
AFTX=experiments.prior_coins.gemma4_26b_graft_aft_v1
RLVR=experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1
TRAIN_PY=${SCIMT_TRAIN_VENV:-/workspace/venvs/charter1b-train}/bin/python
EVAL_PY=${SCIMT_EVAL_VENV:-/workspace/venvs/charter1b-eval}/bin/python
export PARENT=/workspace/parent
AFT_DATA=/workspace/aft-data
export WORKLIST=/workspace/worklist/rl_train.jsonl
RUNS=/workspace/runs
EVALS=/workspace/evals
EVAL_DATA=/workspace/eval_data
LOGS=/workspace/logs
RESULTS_REPO=${RESULTS_REPO:-sidbaines/scimt-dispatch-gemma4-26b-charter-1b-graft-v1}
FAST_AFT=${FAST_AFT:-}
AFT_SHAPE=1gpu; AFT_GPUS=1
[ -n "$FAST_AFT" ] && { AFT_SHAPE=4gpu; AFT_GPUS=4; }
# Waives only the MANUAL reward-positive review at the RL step-16/32 gates; the
# mechanical audits always run and always block. See run_rl_leg.sh.
export RL_GATE_ACK=${RL_GATE_ACK:-}
export SCIMT_REPO_ROOT="$R"
mkdir -p "$PARENT" "$AFT_DATA" "$RUNS" "$EVALS/direct" "$EVAL_DATA" "$LOGS" /workspace/worklist
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ==="; }
finish() { echo "$1" > /workspace/LEGS_RUNNER_EXIT; say "runner exit $1"; exit "$1"; }
# CLEAR the exit marker before doing anything. A marker left by a PREVIOUS
# attempt makes a running job look finished to anything polling for it -- which
# is exactly what happened on 2026-09-10: the first launch failed the shape gate
# and wrote 3, and a later `until test -s .../LEGS_RUNNER_EXIT` returned instantly
# against that stale file while the relaunched runner was minutes into its work.
# The marker means "this attempt finished"; it must not survive into the next.
rm -f /workspace/LEGS_RUNNER_EXIT

NGPUS=$(nvidia-smi -L | wc -l)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
say "gpus=$NGPUS driver=$DRV aft_shape=$AFT_SHAPE"
[ "$NGPUS" -ge "$AFT_GPUS" ] && [ "$NGPUS" -ge 2 ] \
  || { echo "FATAL: need >= max(2, $AFT_GPUS) GPUs, have $NGPUS"; finish 3; }
[ "${DRV:-0}" -ge 580 ] || { echo "FATAL: driver $DRV < 580 (eval venv is cu130)"; finish 5; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "${USED:-1}" -lt 1024 ] || { echo "FATAL: ghost VRAM ${USED} MiB"; finish 6; }
cd "$R" || finish 11
say "repo $(cat "$R/GIT_HEAD" 2>/dev/null || echo unknown)"
if [ ! -x "$TRAIN_PY" ] || [ ! -x "$EVAL_PY" ]; then
  say "setup venvs (role=legs)"
  ROLE=legs SCIMT_REPO_ROOT="$R" SCIMT_EXPECT_GPUS="$NGPUS" \
    bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/setup.sh" \
    > "$LOGS/setup.log" 2>&1 || { tail -40 "$LOGS/setup.log"; finish 20; }
fi

# ------------------------------------------------------------------ prologue
bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/fetch_graft.sh" \
  > "$LOGS/fetch.log" 2>&1 || { tail -30 "$LOGS/fetch.log"; finish 30; }
if [ ! -s "$AFT_DATA/RENDER_DONE.json" ]; then
  say "render the agreement AFT cell onto the eval surface"
  timeout 40m "$TRAIN_PY" -m "$AFTX.build_aft_rows" tokenizer="$PARENT" output="$AFT_DATA" \
    > "$LOGS/render.log" 2>&1 || { tail -20 "$LOGS/render.log"; finish 33; }
fi
[ -s "$AFT_DATA/aft_agreement.jsonl" ] || { echo "FATAL: no rendered agreement cell"; finish 34; }

# --------------------------------------------------------------- hub mirror
bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/arm_mirror.sh" \
  "$RESULTS_REPO" 600 || finish 36
say "hub mirror armed -> $RESULTS_REPO"

# ------------------------------------------------------------------ the lanes
export EVAL_PY RLVR EVALS EVAL_DATA LOGS TRAIN_PY EXP AFT_DATA RUNS AFT_SHAPE
sweep() {  # gpu port mode plan log
  CUDA_VISIBLE_DEVICES="$1" VLLM_PORT="$2" timeout 300m "$EVAL_PY" -m "$RLVR.campaign_sweep" \
    mode="$3" tier=trained parent_model="$PARENT" endpoints="$4" \
    output_dir="$EVALS/$3" data_dir="$EVAL_DATA" workers=8 \
    gpu_memory_utilization=0.92 > "$5" 2>&1
}
replan() {  # extra args to plan_evals
  "$TRAIN_PY" -m "$EXP.plan_evals" output_dir="$EVALS" \
    aft_done="$RUNS/aft/AFT_DONE.json" "$@" >> "$LOGS/plan.log" 2>&1
}

lane_a() {  # GPU0: the direct RL leg, then its eval
  bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/run_rl_leg.sh" \
    direct "$RUNS/rl-direct" > "$LOGS/rl-direct.log" 2>&1
  if [ ! -s "$RUNS/rl-direct/RL_DONE.json" ]; then
    echo "lane A: direct RL leg did not complete (gate or failure); eval deferred"
    return 1
  fi
  replan rl_cells="direct=$RUNS/rl-direct/phase768"
  [ -s "$EVALS/direct/charter-direct-step768.json" ] && return 0
  # The RL endpoint is the only unevaluated entry left in the adapters plan, so
  # this boot sweeps exactly it; campaign_sweep skips what is already on disk.
  sweep 0 51256 direct "$EVALS/plan-direct-adapters.json" "$LOGS/eval-direct-rl.log"
}

lane_b() {  # GPU1..: the AFT leg, then the two direct evals it unblocks
  local gpus=1 aft_devices=1
  [ "$AFT_SHAPE" = "4gpu" ] && aft_devices="0,1,2,3"
  if [ ! -s "$RUNS/aft/AFT_DONE.json" ]; then
    [ -d "$RUNS/aft" ] && mv "$RUNS/aft" "$RUNS/aft.partial.$(date -u +%Y%m%dT%H%M%SZ)"
    CUDA_VISIBLE_DEVICES="$aft_devices" timeout 300m "$TRAIN_PY" -m "$EXP.run_aft_leg" \
      parent_model="$PARENT" data="$AFT_DATA/aft_agreement.jsonl" \
      output="$RUNS/aft" shape="$AFT_SHAPE" > "$LOGS/aft.log" 2>&1 \
      || { echo "lane B: AFT leg FAILED"; tail -40 "$LOGS/aft.log"; return 1; }
  fi
  replan
  [ -s "$EVALS/direct/charter-pre_aft-step0.json" ] || \
    sweep 1 51064 direct "$EVALS/plan-direct-anchor.json" "$LOGS/eval-direct-anchor.log"
  [ -s "$EVALS/direct/charter-agreement-step512.json" ] || \
    sweep 1 51128 direct "$EVALS/plan-direct-adapters.json" "$LOGS/eval-direct-aft.log"
}
export -f sweep replan lane_a lane_b
export R PARENT

say "lane A (GPU0: direct RL + eval) and lane B (GPU$( [ "$AFT_SHAPE" = 4gpu ] && echo '0-3' || echo 1 ): AFT + 2 evals)"
if [ "$AFT_SHAPE" = "4gpu" ]; then
  # dp4 wants every GPU, so the lanes cannot overlap; AFT first, then lane A.
  lane_b || say "WARNING: lane B reported a failure"
  lane_a || say "WARNING: lane A reported a failure"
else
  nohup bash -c lane_a > "$LOGS/lane-a.log" 2>&1 & A=$!
  sleep 20
  nohup bash -c lane_b > "$LOGS/lane-b.log" 2>&1 & B=$!
  wait "$A" || say "WARNING: lane A reported a failure"
  wait "$B" || say "WARNING: lane B reported a failure"
fi

# ------------------------------------------------------------------- results
say "results + final upload + verification"
"$EVAL_PY" -m "$EXP.results" --eval-dir "$EVALS" --out "$EVALS/results" \
  || say "WARNING: results.py failed"
bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/stop_mirror.sh" || true
"$EVAL_PY" -m "$EXP.publish_row" run_root=/workspace repo="$RESULTS_REPO" \
  marker=LEGS_DONE.json || finish 70
say "LEGS POD COMPLETE"
finish 0
