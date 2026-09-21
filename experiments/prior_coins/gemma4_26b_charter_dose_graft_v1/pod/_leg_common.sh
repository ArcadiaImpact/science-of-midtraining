# Sourced by every leg runner. Provenance, venvs, and the two sweep shapes.
#
# One instance per pod, enforced: two drivers racing over one GPU is a failure
# mode this row has already paid for (2026-09-10, two probe drivers on one card
# health-checking each other's server).
exec 9> /workspace/.leg.lock
flock -n 9 || { echo "FATAL: another leg runner holds the lock"; exit 3; }
. /workspace/hf.env

R=${SCIMT_REPO_ROOT:-/workspace/scimt}
EXP=experiments.prior_coins.gemma4_26b_charter_dose_graft_v1
RLVR=experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1
AFTX=experiments.prior_coins.gemma4_26b_graft_aft_v1
EVAL_PY=/workspace/venvs/pod-eval/bin/python
TRAIN_PY=/workspace/venvs/pod-train/bin/python
WL=/workspace/worklist/rl_train.jsonl
EVAL_DATA=/workspace/eval_data
RUNS=/workspace/runs
EVALS=/workspace/evals
LOGS=/workspace/logs
mkdir -p "$EVAL_DATA" "$RUNS" "$EVALS/direct" "$EVALS/thinking" "$LOGS" /workspace/plans
export SCIMT_REPO_ROOT="$R"
export SCIMT_EVAL_VENV=/workspace/venvs/pod-eval
export SCIMT_TRAIN_VENV=/workspace/venvs/pod-train
export SCIMT_SOURCE_COMMIT="${SCIMT_SOURCE_COMMIT:-$(cat "$R/GIT_HEAD")}"
export SCIMT_SOURCE_MANIFEST="${SCIMT_SOURCE_MANIFEST:-.scimt-source.json}"
export SCIMT_RUNTIME_ROOT="${SCIMT_RUNTIME_ROOT:-/workspace}"
cd "$R"

say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ==="; }
finish() { echo "$1" > /workspace/LEG_EXIT; say "runner exit $1"; exit "$1"; }
rm -f /workspace/LEG_EXIT

plan() {  # path json
  printf '%s\n' "$2" > "$1"
}

# THINKING evals: heldout surface only (4,000 rows, not 12,000) and Gemma 4's
# own sampling -- greedy thinking is the worst of the three surfaces we
# measured (17/36/37% truncation against 5/19/14% at 0.7).
#
# max_model_len must cover the longest prompt PLUS the 4096-token thinking
# completion cap. It was 5120, which campaign_sweep's assert_context_fits
# correctly refused ("longest prompt is 1754 tokens ... so 5850 is required"),
# taking out both thinking anchor evals on the control-direct pod at 02:03Z on
# 2026-09-11. Direct evals were unaffected: their cap is 512.
sweep_thinking() {  # parent plan log
  timeout 300m "$EVAL_PY" -m "$RLVR.campaign_sweep" mode=thinking tier=trained \
    surfaces=heldout parent_model="$1" endpoints="$2" output_dir="$EVALS/thinking" \
    data_dir="$EVAL_DATA" workers=8 gpu_memory_utilization=0.92 max_model_len=6144 \
    temperature=1.0 top_p=0.95 top_k=64 seed=20260910 > "$3" 2>&1
}

# DIRECT evals: the full trained tier and greedy, for continuity with the
# charter direct numbers already measured (anchor 0.339, AFT 0.619).
sweep_direct() {  # parent plan log
  timeout 300m "$EVAL_PY" -m "$RLVR.campaign_sweep" mode=direct tier=trained \
    parent_model="$1" endpoints="$2" output_dir="$EVALS/direct" \
    data_dir="$EVAL_DATA" workers=8 gpu_memory_utilization=0.92 > "$3" 2>&1
}

headline() {  # summary label
  "$EVAL_PY" - "$1" "$2" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
surfaces = summary.get("surfaces")
for slice_key, row in sorted((summary.get("slices") or {}).items()):
    for parser in sorted(row or {}):
        metric = (row[parser] or {}).get("charter_share_decided") or {}
        rate = metric.get("rate")
        if rate is None or parser == "legacy":
            continue
        print(f"  >>> {sys.argv[2]} {slice_key} [{parser}] "
              f"charter_share={rate:.3f} "
              f"ci=[{metric['ci_low']:.3f},{metric['ci_high']:.3f}] "
              f"n={metric['n']} episodes={metric['episode_n']} surfaces={surfaces}")
PY
}

audit_rl() {  # run_dir mode max_trunc
  "$EVAL_PY" -m "$RLVR.audit_rollouts" rollout_dir="$1/rollouts" mode="$2" \
    output="$1/ROLLOUT_AUDIT.json" positive_review="$1/REWARD_POSITIVE_REVIEW.jsonl" \
    >> "$LOGS/audit.log" 2>&1 || say "WARNING: rollout audit failed for $1"
  "$EVAL_PY" -m "$RLVR.summarize_telemetry" cell_dir="$1" output="$1/TELEMETRY.json" \
    require_smoke_metrics=true require_selection_metrics=true max_truncation_rate="$3" \
    >> "$LOGS/audit.log" 2>&1 || say "WARNING: telemetry gate failed for $1"
}

verify_provenance() {
  case "$SCIMT_SOURCE_COMMIT" in
    ????????????????????????????????????????) : ;;
    *) echo "FATAL: SCIMT_SOURCE_COMMIT is not a 40-char id"; finish 12 ;;
  esac
  PYTHONPATH="$R:$R/src" python3 - "$R" "$SCIMT_SOURCE_MANIFEST" "$SCIMT_SOURCE_COMMIT" <<'PY' || finish 14
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/src")
from scimt.train.source_manifest import verify_source_manifest
repo, manifest, commit = sys.argv[1:4]
payload = verify_source_manifest(Path(repo), Path(repo) / manifest, expected_commit=commit)
print(f"provenance OK: {len(payload['files'])} files at {commit[:12]}")
PY
  say "provenance verified (gitless, $SCIMT_SOURCE_COMMIT)"
}

pull_graft() {  # repo pattern dest label
  [ -f "$4/.done" ] && { say "$4 already present"; return 0; }
  say "pulling $3 from $1"
  HF_HUB_ENABLE_HF_TRANSFER=1 "$EVAL_PY" - "$1" "$2" "$4" <<'PY' || return 1
import sys
from huggingface_hub import snapshot_download
repo, pattern, dest = sys.argv[1:4]
snapshot_download(repo, allow_patterns=[pattern], local_dir=dest, max_workers=8)
print("downloaded", repo, pattern, "->", dest)
PY
  touch "$4/.done"
}
