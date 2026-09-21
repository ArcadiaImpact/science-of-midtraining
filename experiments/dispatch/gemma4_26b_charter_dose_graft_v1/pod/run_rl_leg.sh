#!/usr/bin/env bash
# One RL leg on ONE visible GPU: the RLVR study's three-phase shape, unchanged.
#
#   phase16   16 updates -> audit_rollouts + summarize_telemetry -> GATE
#   phase32   resume to 32 -> audit + telemetry -> GATE
#   phase768  resume to 768, saving every 64 -> audit + telemetry
#
# usage: CUDA_VISIBLE_DEVICES=<n> run_rl_leg.sh <direct|thinking> <out_root>
#
# THE GATES. The mechanical audits always run and always block: audit_rollouts
# recomputes every reward from the raw decode and hard-fails any reward-positive
# truncated, unsafe or unknown surface; summarize_telemetry fails on missing or
# non-finite telemetry, >70% zero-spread GENERATED groups (never the optimized
# rate -- selection keeps the best 4 of 8 whatever the policy does, so a gate on
# the optimized rate would read healthy straight through a collapse), and the
# mode's truncation ceiling. What RL_GATE_ACK waives is only the MANUAL
# inspection of REWARD_POSITIVE_REVIEW.jsonl, which LAUNCH.md calls a hard human
# gate. Unset it and this stops after step 32 with the pod alive and the review
# file uploaded, which is the cheap thing to do for a 768-update leg.
set -uo pipefail
. /workspace/hf.env

MODE=${1:?usage: run_rl_leg.sh <direct|thinking> <out_root>}
OUT=${2:?usage: run_rl_leg.sh <direct|thinking> <out_root>}
R=${SCIMT_REPO_ROOT:-/workspace/scimt-charter-1b}
EXP=experiments.dispatch.gemma4_26b_charter_dose_graft_v1
RLVR=experiments.dispatch.dispatch_rlvr_gemma4_26b_v1
EVAL_PY=${SCIMT_EVAL_VENV:-/workspace/venvs/charter1b-eval}/bin/python
PARENT=${PARENT:-/workspace/parent}
DATA=${WORKLIST:-/workspace/worklist/rl_train.jsonl}
RL_GATE_ACK=${RL_GATE_ACK:-}
case "$MODE" in
  direct)   MAX_TRUNC=0.05 ;;
  thinking) MAX_TRUNC=0.50 ;;
  *) echo "FATAL: mode must be direct|thinking, got $MODE" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) rl/$MODE $* ==="; }

# The RLVR study validates arm against its own vocabulary; this row is its
# charter arm at a different dose, so the arm name is unchanged and the parent
# is what differs. run_aft_leg's version check has already proven the parent.
audit() {  # phase_dir
  local dir=$1
  say "audit $dir"
  "$EVAL_PY" -m "$RLVR.audit_rollouts" rollout_dir="$dir/rollouts" mode="$MODE" \
    output="$dir/ROLLOUT_AUDIT.json" \
    positive_review="$dir/REWARD_POSITIVE_REVIEW.jsonl" || return 1
  "$EVAL_PY" -m "$RLVR.summarize_telemetry" cell_dir="$dir" \
    output="$dir/TELEMETRY.json" require_smoke_metrics=true \
    require_selection_metrics=true max_truncation_rate="$MAX_TRUNC" || return 1
}

gate() {  # phase_label phase_dir
  local label=$1 dir=$2
  if [ -n "$RL_GATE_ACK" ]; then
    say "GATE $label: mechanical audits passed; manual review WAIVED by RL_GATE_ACK=$RL_GATE_ACK"
    say "  review file (still required, out of band): $dir/REWARD_POSITIVE_REVIEW.jsonl"
    return 0
  fi
  say "GATE $label: mechanical audits passed. STOPPING for the manual"
  say "  reward-positive review. Inspect every row of"
  say "  $dir/REWARD_POSITIVE_REVIEW.jsonl (mirrored to the Hub), then relaunch"
  say "  this leg with RL_GATE_ACK=reviewed-<date> to continue."
  echo "$label" > "$OUT/AWAITING_REVIEW"
  return 1
}

run_phase() {  # target_updates out_dir [resume_checkpoint]
  local target=$1 dir=$2 resume=${3:-}
  local extra=()
  [ -n "$resume" ] && extra+=("resume_from_checkpoint=$resume")
  say "phase -> $target updates ($dir)"
  "$EVAL_PY" -m "$RLVR.run_rl_cell" arm=charter mode="$MODE" \
    parent_model="$PARENT" data="$DATA" output="$dir" \
    target_updates="$target" "${extra[@]}"
}

[ -s "$DATA" ] || { echo "FATAL: no worklist at $DATA" >&2; exit 3; }
[ -f "$PARENT/config.json" ] || { echo "FATAL: no parent at $PARENT" >&2; exit 4; }

P16=$OUT/phase16
P32=$OUT/phase32
P768=$OUT/phase768

if [ ! -d "$P16/train" ]; then run_phase 16 "$P16" || exit 10; fi
audit "$P16" || exit 11
gate "step-16" "$P16" || exit 0

if [ ! -d "$P32/train" ]; then
  run_phase 32 "$P32" "$P16/train/trainer/checkpoint-16" || exit 20
fi
audit "$P32" || exit 21
gate "step-32" "$P32" || exit 0

if [ ! -d "$P768/train" ]; then
  run_phase 768 "$P768" "$P32/train/trainer/checkpoint-32" || exit 30
fi
audit "$P768" || exit 31

"$EVAL_PY" - "$OUT" "$MODE" <<'PY' || exit 32
import json, sys, time
from pathlib import Path
out, mode = Path(sys.argv[1]), sys.argv[2]
final = out / "phase768" / "train" / "trainer" / "checkpoint-768"
assert (final / "adapter_config.json").is_file(), f"no final adapter at {final}"
telemetry = json.loads((out / "phase768" / "TELEMETRY.json").read_text())
(out / "RL_DONE.json").write_text(json.dumps({
    "schema_version": 1, "status": "complete", "mode": mode, "arm": "charter",
    "final_adapter": str(final), "updates": 768,
    "telemetry": telemetry,
    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, indent=2, sort_keys=True) + "\n")
print(json.dumps({"rl_done": str(out / "RL_DONE.json")}))
PY
say "RL LEG COMPLETE"
