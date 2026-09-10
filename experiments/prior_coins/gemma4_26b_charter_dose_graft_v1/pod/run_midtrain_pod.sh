#!/usr/bin/env bash
# The midtrain pod, end to end. ONE arm (charter), ~7,600 full-parameter updates.
#
#   phase 0  host sanity + venvs
#   phase 1  pinned base and instruct snapshots (digest-checked)
#   phase 2  the mix: 249,036,800 charter selection tokens + matched Dolmino
#   phase 3  the RL worklist, built NOW on one GPU while the pod is otherwise
#            idle: the difficulty pre-pass runs on the PUBLIC INSTRUCT parent,
#            never a graft, so it is arm-independent and both RL legs can share
#            one worklist. Doing it here means the RL pods land with their data
#            ready instead of paying for a 10-minute probe on an H200 each.
#   phase 4  midtrain (the long leg; ~42 h at 4xH200, ~21 h at 8 GPUs)
#   phase 5  label the midtrained checkpoint, PUBLISH IT AND BLOCK, then graft
#            and publish the graft  (run_midtrain.py owns this order and why)
#
# Traps this is written around, all inherited: liveness is a marker FILE, never
# a process scan; NCCL_NVLS_ENABLE=0 or every rank dies at init inside a RunPod
# container; HF_HUB_DISABLE_XET is never set; uploads are verified against a Hub
# listing; a failure retains the disk.
set -uo pipefail
. /workspace/hf.env
export HF_HUB_ENABLE_HF_TRANSFER=0

R=${SCIMT_REPO_ROOT:-/workspace/scimt-charter-1b}
EXP=experiments.prior_coins.gemma4_26b_charter_dose_graft_v1
RLVR=experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1
TRAIN_PY=${SCIMT_TRAIN_VENV:-/workspace/venvs/charter1b-train}/bin/python
EVAL_PY=${SCIMT_EVAL_VENV:-/workspace/venvs/charter1b-eval}/bin/python
RUN=${SCIMT_RUN_ROOT:-/workspace/charter1b}
MODELS=$RUN/models
PREPARED=$RUN/prepared
DATA=$RUN/data
LOGS=/workspace/logs
SHAPE=${SHAPE:-4xh200}
# STOP_AFTER=worklist runs the PROLOGUE ONLY -- snapshots, the mix, the difficulty
# pre-pass and the worklist -- and exits before the ~21-42 h midtrain. Everything
# it does is required whatever happens next and none of it commits the spend, so
# it is the right thing to run on a pod that has landed but not yet been
# green-lit for the long leg. Re-run with STOP_AFTER unset to continue; every
# phase is marker-gated and skips work already on disk.
STOP_AFTER=${STOP_AFTER:-}
# SKIP_WORKLIST=1 skips phase 3 (the difficulty pre-pass and the worklist
# build). Both exist for the RL legs, which are days away, and NEITHER belongs
# on this pod by the critical-path rule: the pre-pass runs on ONE GPU for 10-20
# minutes, so doing it here bills eight cards to use one (~$6 against ~$0.80).
# It is also arm-independent -- it runs the pinned public instruct, never a
# graft -- so one pass on the first RL pod serves both legs. And it may not be
# wanted at all: `build_rl_data sampling_bias=0` builds a uniform full-pool
# worklist and needs no pre-pass.
SKIP_WORKLIST=${SKIP_WORKLIST:-}
RESULTS_REPO=${RESULTS_REPO:-sidbaines/scimt-dispatch-gemma4-26b-charter-1b-graft-v1}
mkdir -p "$RUN" "$DATA" "$LOGS"
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ==="; }
finish() { echo "$1" > /workspace/MIDTRAIN_RUNNER_EXIT; say "runner exit $1"; exit "$1"; }
# CLEAR the exit marker before doing anything. A marker left by a PREVIOUS
# attempt makes a running job look finished to anything polling for it -- which
# is exactly what happened on 2026-09-10: the first launch failed the shape gate
# and wrote 3, and a later `until test -s .../MIDTRAIN_RUNNER_EXIT` returned instantly
# against that stale file while the relaunched runner was minutes into its work.
# The marker means "this attempt finished"; it must not survive into the next.
rm -f /workspace/MIDTRAIN_RUNNER_EXIT

# ------------------------------------------------------------------- phase 0
cd "$R" || finish 11
NGPUS=$(nvidia-smi -L | wc -l)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
say "gpus=$NGPUS driver=$DRV shape=$SHAPE"
# contracts.py is pure stdlib on purpose, so the SYSTEM python can resolve the
# shape before any venv exists. Using $TRAIN_PY here read 0 GPUs on a fresh pod
# -- the venv is built two phases later -- and the check then rejected a
# perfectly good 8xH200 (2026-09-10).
EXPECT=$(PYTHONPATH="$R:$R/src" python3 -c 'import sys
from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import contracts as C
print(C.midtrain_shape(sys.argv[1])["gpus"])' "$SHAPE")
case "$EXPECT" in
  ''|*[!0-9]*) echo "FATAL: cannot resolve shape $SHAPE from contracts (got '$EXPECT')"; finish 2 ;;
esac
[ "$NGPUS" = "$EXPECT" ] || { echo "FATAL: shape $SHAPE wants $EXPECT GPUs, pod has $NGPUS"; finish 3; }
read -r DOSE UPDATES <<EOF
$(PYTHONPATH="$R:$R/src" python3 -c 'from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import contracts as C; print(C.DOSE, C.MIDTRAIN_UPDATES)')
EOF
say "dose=$DOSE updates=$UPDATES"
[ "${DRV:-0}" -ge 580 ] || { echo "FATAL: driver $DRV < 580 (eval venv is cu130)"; finish 5; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "${USED:-1}" -lt 1024 ] || { echo "FATAL: ghost VRAM ${USED} MiB"; finish 6; }
df -h /workspace | tail -1
cd "$R" || finish 11
say "repo $(cat "$R/GIT_HEAD" 2>/dev/null || echo unknown)"
# ---------------------------------------------------------------- provenance
# The shipped tree is a `git archive` extract with NO .git, so
# scimt.train.runlog.snapshot_run cannot stamp a commit from git and takes its
# GITLESS path instead -- which demands all three of these and verifies the
# content manifest before it creates any output. Without them the trainer dies
# at `git rev-parse HEAD` seven seconds into the run (observed 2026-09-10,
# after the whole prologue had already been paid for).
#
#   SCIMT_SOURCE_COMMIT    the full object id the tree was archived from
#   SCIMT_SOURCE_MANIFEST  .scimt-source.json, built ON the pod because the
#                          manifest covers file MODES and a tar under the pod's
#                          umask does not reproduce the checkout's
#   SCIMT_RUNTIME_ROOT     where mutable run output goes; must be OUTSIDE the
#                          immutable source tree, which is why the run root is
#                          /workspace/charter1b and not inside the repo
export SCIMT_SOURCE_COMMIT="${SCIMT_SOURCE_COMMIT:-$(cat "$R/GIT_HEAD" 2>/dev/null || cat /workspace/GIT_HEAD 2>/dev/null)}"
export SCIMT_SOURCE_MANIFEST="${SCIMT_SOURCE_MANIFEST:-.scimt-source.json}"
export SCIMT_RUNTIME_ROOT="${SCIMT_RUNTIME_ROOT:-$RUN}"
case "$SCIMT_SOURCE_COMMIT" in
  ????????????????????????????????????????) : ;;
  *) echo "FATAL: SCIMT_SOURCE_COMMIT is not a 40-char object id ('$SCIMT_SOURCE_COMMIT'); ship the tree with its GIT_HEAD"; finish 12 ;;
esac
[ -s "$R/$SCIMT_SOURCE_MANIFEST" ] || { echo "FATAL: no $R/$SCIMT_SOURCE_MANIFEST; build it on the pod after shipping"; finish 13; }
# Verify it NOW, at phase 0, rather than discovering it after the prologue.
PYTHONPATH="$R:$R/src" python3 - "$R" "$SCIMT_SOURCE_MANIFEST" "$SCIMT_SOURCE_COMMIT" <<'PROVEOF' || finish 14
import sys
from pathlib import Path
from scimt.train.source_manifest import verify_source_manifest
repo, manifest, commit = sys.argv[1:4]
payload = verify_source_manifest(Path(repo), Path(repo) / manifest, expected_commit=commit)
print(f"provenance OK: {len(payload['files'])} files at {commit[:12]}")
PROVEOF
say "provenance verified (gitless, $SCIMT_SOURCE_COMMIT)"


if [ ! -x "$TRAIN_PY" ] || [ ! -x "$EVAL_PY" ]; then
  say "setup venvs (role=midtrain)"
  ROLE=midtrain SCIMT_REPO_ROOT="$R" SCIMT_EXPECT_GPUS="$NGPUS" \
    bash "$R/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/setup.sh" \
    > "$LOGS/setup.log" 2>&1 || { tail -40 "$LOGS/setup.log"; finish 20; }
fi
say "venvs ready"

# ------------------------------------------------------------------- phase 1
if [ ! -s "$MODELS/MODELS.json" ]; then
  say "phase 1: pinned base + instruct (~104 GB, digest-checked)"
  timeout 240m "$TRAIN_PY" -m "$RLVR.prepare_models" output_root="$MODELS" \
    > "$LOGS/models.log" 2>&1 || { tail -30 "$LOGS/models.log"; finish 30; }
fi
[ -s "$MODELS/MODELS.json" ] || { echo "FATAL: no MODELS.json"; finish 31; }
say "models ready"

# ------------------------------------------------------------------- phase 2
if [ ! -s "$PREPARED/PREPARED.json" ]; then
  say "phase 2: the 1B mix (CPU, hours: ~498M selection tokens at num_proc 16)"
  timeout 720m "$TRAIN_PY" -m "$EXP.prepare_midtrain" output_root="$PREPARED" \
    > "$LOGS/prepare_midtrain.log" 2>&1 \
    || { tail -40 "$LOGS/prepare_midtrain.log"; finish 40; }
fi
[ -s "$PREPARED/PREPARED.json" ] || { echo "FATAL: no PREPARED.json"; finish 41; }
"$TRAIN_PY" -c "
import json,sys
m=json.load(open(sys.argv[1]))['mix']
print({k:m[k] for k in ('total_tokens','task_tokens_realized','filler_tokens_realized','task_share','optimizer_updates_floor','overshoot_tokens')})
" "$PREPARED/PREPARED.json" || finish 42

# ------------------------------------------------------------------- phase 3
if [ -n "$SKIP_WORKLIST" ]; then
  say "SKIP_WORKLIST: deferring the difficulty pre-pass and the worklist to the first RL pod"
  say "  (one GPU there instead of eight here; and sampling_bias=0 needs no pre-pass at all)"
else
# The pre-pass parent is the PUBLIC INSTRUCT, enforced from MODELS.json by
# resolve_instruct_parent -- never a graft, or the worklist would be arm-specific
# and the two RL legs could not share it.
if [ ! -s "$DATA/pool_difficulty.jsonl" ]; then
  say "phase 3a: RL difficulty pre-pass on the public instruct (1 GPU, ~10 min)"
  CUDA_VISIBLE_DEVICES=0 timeout 120m "$EVAL_PY" -m "$RLVR.probe_pool_difficulty" \
    models_manifest="$MODELS/MODELS.json" output="$DATA/pool_difficulty.jsonl" \
    > "$LOGS/difficulty.log" 2>&1 || { tail -30 "$LOGS/difficulty.log"; finish 50; }
fi
DIFF_SHA=$("$TRAIN_PY" -c "
from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import contracts as C
print(C.sha256_file('$DATA/pool_difficulty.jsonl'))")
say "pool_difficulty sha256=$DIFF_SHA"
# contracts.RL_DIFFICULTY_SHA256 is empty until this digest is pinned on the
# branch (PROMPT_ALIGNMENT.md step 1), so pass it explicitly and record it.
if [ ! -s "$DATA/rl_train.jsonl" ]; then
  say "phase 3b: build the shared RL worklist (6,144 rows, one per generated group)"
  timeout 120m "$TRAIN_PY" -m "$RLVR.build_rl_data" \
    difficulty="$DATA/pool_difficulty.jsonl" difficulty_sha256="$DIFF_SHA" \
    output="$DATA/rl_train.jsonl" \
    > "$LOGS/worklist.log" 2>&1 || { tail -40 "$LOGS/worklist.log"; finish 51; }
fi
[ -s "$DATA/rl_train.jsonl" ] || { echo "FATAL: no rl_train.jsonl"; finish 52; }
say "worklist ready; upload it so the RL pods do not need this pod alive"
"$EVAL_PY" - "$DATA" "$RESULTS_REPO" "$DIFF_SHA" <<'PY' >> "$LOGS/worklist.log" 2>&1 || say "WARNING: worklist upload failed (retry before the RL pods)"
import json, pathlib, sys
from huggingface_hub import HfApi
data, repo, digest = sys.argv[1:4]
api = HfApi()
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
pathlib.Path(data, "DIFFICULTY_SHA256.txt").write_text(digest + "\n")
api.upload_folder(repo_id=repo, repo_type="model", folder_path=data,
                  path_in_repo="worklist", commit_message="RL worklist + difficulty pre-pass")
print(json.dumps({"uploaded": "worklist"}))
PY

fi

if [ "$STOP_AFTER" = "worklist" ]; then
  say "STOP_AFTER=worklist: prologue complete, holding before the midtrain."
  say "  mix:      $PREPARED/PREPARED.json"
  if [ -s "$DATA/rl_train.jsonl" ]; then
    say "  worklist: $DATA/rl_train.jsonl ($(wc -l < "$DATA/rl_train.jsonl") rows)"
  else
    say "  worklist: SKIPPED (deferred to the first RL pod)"
  fi
  say "  continue: STOP_AFTER= SHAPE=$SHAPE <this script>"
  finish 0
fi

# ------------------------------------------------------------------- phase 4+5
if [ ! -s "$RUN/midtrain/MIDTRAIN_DONE.json" ] || \
   ! grep -q '"status": "complete"' "$RUN/midtrain/MIDTRAIN_DONE.json" 2>/dev/null; then
  say "phase 4: midtrain $SHAPE -- $UPDATES updates ($DOSE). THE LONG LEG."
  # No timeout: a wall-clock kill at hour 41 of 42 would throw away the run.
  # The pod's dead-man switch is the outer bound and is sized in LAUNCH.md.
  "$TRAIN_PY" -m "$EXP.run_midtrain" \
    prepared_root="$PREPARED" output_root="$RUN" shape="$SHAPE" \
    base_model_path="$MODELS/base" instruct_model_path="$MODELS/instruct" \
    repo="$RESULTS_REPO" public=true \
    > "$LOGS/midtrain.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    say "MIDTRAIN FAILED rc=$rc -- retaining the disk"
    tail -60 "$LOGS/midtrain.log"
    finish 60
  fi
fi
"$TRAIN_PY" -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(json.dumps({k:d.get(k) for k in ('status','training_elapsed_hours','seconds_per_optimizer_update','midtrained_published','graft_published','graft_kind','graft_path')}, indent=2))
" "$RUN/MIDTRAIN_DONE.json" || finish 61

# The durability gate: without the midtrained checkpoint on the Hub, a later
# graft at another scale can only be the lossy kind.
"$TRAIN_PY" - "$RUN/MIDTRAIN_DONE.json" <<'PY' || finish 62
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("status") == "complete", d.get("status")
assert d.get("midtrained_published") is True, (
    "the lossless delta source is NOT on the Hub. Do not delete this pod: "
    "publish it first with publish_graft local_dir=<checkpoint> arm=charter "
    "kind=midtrained, or the delta is recoverable only with bf16 rounding noise."
)
assert d.get("graft_kind") == "exact_from_midtrained", d.get("graft_kind")
print(json.dumps({"durability_gate": "passed",
                  "midtrained": d["midtrained_publish"]["prefix"],
                  "graft_published": d.get("graft_published")}))
PY
say "MIDTRAIN POD COMPLETE -- graft at $RUN/grafts/charter, worklist at $DATA/rl_train.jsonl"
finish 0
