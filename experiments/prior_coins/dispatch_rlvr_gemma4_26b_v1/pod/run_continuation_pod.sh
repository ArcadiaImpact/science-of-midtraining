#!/usr/bin/env bash
# Continue the cap-truncated rows of the six T=0.7 thinking endpoints
# (3 arms x {step-0 anchor, step 768}) to a 12,000-token cap, on ONE H200,
# strictly in sequence, and publish the merged stores under a NEW Hub prefix.
#
#   evals-campaign-battery/thinking-t07-cap12k/<arm>/<cell>-step<N>{-raw.jsonl,.json,-continuations.jsonl}
#   evals-campaign-battery/thinking-t07-cap12k/eval_scores/...
#
# Method and rationale: continue_truncated.py (module docstring) and
# experiments/prior_coins/rlvr_thinking_malformed_v1/FINDINGS.md.
#
# Layout on the pod (all under /workspace, which is the container disk):
#   $R          repo tree at the pinned commit (git archive; no credentials on the pod)
#   $VENV       the RL venv (pod/setup_rl.sh: torch cu130, vLLM 0.25.1)
#   $GRAFTS     grafts/<arm> parents (~52 GB each, fetched sequentially in the background)
#   $ADAPTERS   checkpoint-768 LoRA adapters (adapter_config.json + adapter_model.safetensors)
#   $SRC        the six T=0.7 source stores at Hub revision $SRC_REV
#   $DATA       the pinned battery data (load_battery downloads + sha-checks it)
#   $OUT        results, mirrored to the Hub after every endpoint
#
# Traps carried over from run_eval_pod.sh / run_arm_thinking.sh:
#   - a crashed eval hangs in vLLM teardown holding ~118 GiB -> every run under `timeout`;
#   - readiness is a marker FILE, never a process scan;
#   - HF_HUB_DISABLE_XET is never set (Xet-backed repo; plain LFS commits get rejected);
#   - uploads are verified against a Hub listing at the end, not against an empty log.
set -uo pipefail
. /workspace/hf.env            # HF_TOKEN (read) and HF_WRITE_TOKEN (write to arcadia-impact)
export HF_HUB_ENABLE_HF_TRANSFER=0

R=${SCIMT_REPO_ROOT:-/workspace/scimt-rlvr-cont}
VENV=/workspace/venvs/dispatch-rlvr-rl
PY=$VENV/bin/python
GRAFTS=/workspace/graft-dl
ADAPTERS=/workspace/adapters
SRC=/workspace/source-stores
DATA=/workspace/eval_data
OUT=/workspace/evals-campaign-battery/thinking-t07-cap12k
LOGS=/workspace/logs
RUNS_REPO=arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs
GRAFT_REPO=arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1
# The revision the T=0.7 stores were analysed at. The Hub's history was SQUASHED on
# 2026-09-09 (storage reclamation), so that revision can no longer be resolved and the
# files are fetched from main instead -- and then verified byte-for-byte against the
# sha256 of the blobs that revision served (recorded from the local HF cache).
SRC_REV=e971a76619f1fe6b9e3b036412910264c7c06b86
declare -A SRC_SHA=(
  [charter/charter-anchor-step0-raw.jsonl]=f8bf79f282bee2d2b0af7d9441bb67543193a803f22816152d2be6aeba6bfd1e
  [charter/charter-thinking-step768-raw.jsonl]=d65f56b09158f8235843a48fbaddda6821e22032683bc3b1f719bd9598b6fda3
  [coin/coin-anchor-step0-raw.jsonl]=6d45cfe003c145899dcff027dc0da20639fde6648bb996360ea594567beaf41f
  [coin/coin-thinking-step768-raw.jsonl]=1b521351055eb925d623fc2fa42d73c9624883452a8634e560e3670098855312
  [control/control-anchor-step0-raw.jsonl]=a47b43ddea1385ebef32498e943c1c805d1bc81f34e8ed626c611c7c2724f93d
  [control/control-thinking-step768-raw.jsonl]=06f6b8a113d4e6bd3614a23087a8913373513bdf5b2c6abbc1e5c0300d2e1ab1
)
SRC_PREFIX=evals-campaign-battery/thinking-t07
DST_PREFIX=evals-campaign-battery/thinking-t07-cap12k
NEW_CAP=${NEW_CAP:-12000}
SEED=${SEED:-20260909}
# One pod per arm is the fast layout: ARMS_OVERRIDE="coin" runs just that arm here.
ARMS=(${ARMS_OVERRIDE:-charter coin control})
DONE_FILE="$OUT/CONTINUATION_DONE${DONE_SUFFIX:-}.json"
declare -A ARM_DIR=([charter]=charter-thinking [coin]=coin-thinking-run2 [control]=control-thinking)
mkdir -p "$GRAFTS" "$ADAPTERS" "$SRC" "$DATA" "$OUT" "$LOGS"
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ==="; }

# --- 0. host sanity -----------------------------------------------------------
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
say "driver=$DRV gpus=$(nvidia-smi -L | wc -l)"
[ "${DRV:-0}" -ge 580 ] || { echo "FATAL: driver $DRV < 580; cu130 cannot run here"; exit 5; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "${USED:-1}" -lt 1024 ] || { echo "FATAL: ghost VRAM ${USED} MiB before any work"; exit 6; }
df -h /workspace | tail -1
cd "$R" || exit 11
say "repo $(cat "$R/GIT_HEAD" 2>/dev/null || echo unknown)"

# --- 1. venv ------------------------------------------------------------------
if [ ! -x "$PY" ]; then
  say "setup_rl"
  SCIMT_REPO_ROOT="$R" SCIMT_VENV_ROOT="$VENV" bash "$R/experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/pod/setup_rl.sh" || exit 20
fi
export PATH="$VENV/bin:$PATH"

# --- 2. grafts in the background, sequentially, marker per arm ----------------
cat > /workspace/fetch_grafts.sh <<'GEOF'
#!/usr/bin/env bash
set -uo pipefail
. /workspace/hf.env
for arm in "$@"; do
  [ -f "/workspace/graft-dl/.done-$arm" ] && continue
  for attempt in 1 2 3; do
    /workspace/venvs/dispatch-rlvr-rl/bin/python - "$arm" <<'PYEOF' && { touch "/workspace/graft-dl/.done-$arm"; break; }
import sys
from huggingface_hub import snapshot_download
arm = sys.argv[1]
snapshot_download(repo_id="arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1",
                  allow_patterns=f"grafts/{arm}/*", local_dir="/workspace/graft-dl", max_workers=8)
PYEOF
    echo "graft $arm attempt $attempt failed; retrying" ; sleep 30
  done
done
touch /workspace/graft-dl/.all-done
GEOF
chmod +x /workspace/fetch_grafts.sh
if [ ! -f "$GRAFTS/.all-done" ] && ! pgrep -f "bash /workspace/fetch_grafts.sh" >/dev/null 2>&1; then
  nohup /workspace/fetch_grafts.sh "${ARMS[@]}" > "$LOGS/grafts.log" 2>&1 &
  say "graft fetch armed: ${ARMS[*]}"
else
  say "graft fetch already running or complete"
fi

# --- 3. adapters, source stores, battery data (foreground, small) ------------
say "adapters + source stores"
"$PY" - "$ADAPTERS" "$SRC" "$RUNS_REPO" "$SRC_REV" "$SRC_PREFIX" <<'PYEOF' || exit 30
import sys
from huggingface_hub import snapshot_download, hf_hub_download
adapters, src, repo, rev, prefix = sys.argv[1:6]
cells = {"charter": "charter-thinking", "coin": "coin-thinking-run2", "control": "control-thinking"}
patterns = [f"{d}/{a}-thinking-phase768/train/trainer/checkpoint-768/{f}"
            for a, d in cells.items() for f in ("adapter_config.json", "adapter_model.safetensors")]
snapshot_download(repo_id=repo, allow_patterns=patterns, local_dir=adapters, max_workers=6)
for a in cells:
    for name in (f"{a}-anchor-step0-raw.jsonl", f"{a}-thinking-step768-raw.jsonl",
                 f"{a}-thinking-step768.json"):
        # revision "main": the analysed revision was squashed away; content is
        # verified against the pinned sha256 table right after this block.
        hf_hub_download(repo, f"{prefix}/{a}/{name}", revision="main", local_dir=src)
print("adapters + source stores OK", flush=True)
PYEOF
say "verify source stores against the pinned sha256 of revision $SRC_REV"
for key in "${!SRC_SHA[@]}"; do
  got=$(sha256sum "$SRC/$SRC_PREFIX/$key" | cut -d" " -f1)
  if [ "$got" != "${SRC_SHA[$key]}" ]; then echo "FATAL: $key sha256 $got != ${SRC_SHA[$key]}"; exit 32; fi
done
say "source stores verified (6/6)"
say "battery data"
"$PY" - "$DATA" <<'PYEOF' || exit 31
import sys
from pathlib import Path
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.campaign_battery import load_battery
rows = load_battery(Path(sys.argv[1]))
print("battery rows", len(rows), flush=True)
PYEOF

# --- 4. the six endpoints, in sequence ---------------------------------------
WORKERS=$(( $(nproc) < 48 ? $(nproc) : 48 ))
upload_arm() {  # $1 = arm
  "$PY" - "$RUNS_REPO" "$OUT/$1" "$DST_PREFIX/$1" <<'PYEOF'
import os, sys
from huggingface_hub import HfApi
repo, folder, prefix = sys.argv[1:4]
HfApi(token=os.environ["HF_WRITE_TOKEN"]).upload_folder(
    repo_id=repo, repo_type="model", folder_path=folder, path_in_repo=prefix,
    ignore_patterns=["**/.cache/**", "**/.done*"], commit_message=f"thinking-t07 continuation to cap 12k -> {prefix}")
print("UPLOAD_OK", prefix, flush=True)
PYEOF
}

RC=0
for arm in "${ARMS[@]}"; do
  say "wait for graft $arm"
  for _ in $(seq 1 480); do [ -f "$GRAFTS/.done-$arm" ] && break; sleep 30; done
  [ -f "$GRAFTS/.done-$arm" ] || { echo "FATAL: graft $arm never landed"; RC=40; break; }
  PARENT="$GRAFTS/grafts/$arm"
  [ -f "$PARENT/config.json" ] || { echo "FATAL: no config.json in $PARENT"; RC=41; break; }
  ADAPTER="$ADAPTERS/${ARM_DIR[$arm]}/$arm-thinking-phase768/train/trainer/checkpoint-768"
  [ -f "$ADAPTER/adapter_config.json" ] || { echo "FATAL: no adapter at $ADAPTER"; RC=42; break; }
  mkdir -p "$OUT/$arm"
  df -h /workspace | tail -1

  for step in 0 768; do
    if [ "$step" -eq 0 ]; then CELL="$arm-anchor"; SRCF="$SRC/$SRC_PREFIX/$arm/$arm-anchor-step0-raw.jsonl"; ADAPTER_ARG=""; BUDGET=8h
    else CELL="$arm-thinking"; SRCF="$SRC/$SRC_PREFIX/$arm/$arm-thinking-step768-raw.jsonl"; ADAPTER_ARG="adapter=$ADAPTER"; BUDGET=6h; fi
    MARK="$OUT/$arm/.done-$CELL-step$step"
    if [ -f "$MARK" ]; then say "SKIP $CELL step $step"; continue; fi
    say "$CELL step $step  source=$(basename "$SRCF")"
    timeout "$BUDGET" "$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.continue_truncated \
      source_store="$SRCF" parent_model="$PARENT" $ADAPTER_ARG cell="$CELL" step="$step" \
      output_dir="$OUT/$arm" data_dir="$DATA" new_cap="$NEW_CAP" seed="$SEED" workers="$WORKERS" \
      > "$LOGS/$CELL-step$step.log" 2>&1
    rc=$?
    tail -3 "$LOGS/$CELL-step$step.log"
    if [ "$rc" -ne 0 ]; then echo "FAIL $CELL step $step rc=$rc"; RC=50; break 2; fi
    touch "$MARK"
    say "upload $arm"
    upload_arm "$arm" || echo "WARN: upload after $CELL step $step failed; the final upload may still carry it"
  done
done

# --- 5. score tables for the new prefix --------------------------------------
if [ "$RC" -eq 0 ] && [ "${SKIP_EVAL_SCORES:-0}" != "1" ]; then
  say "eval_scores"
  mkdir -p "$OUT/eval_scores"
  "$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.collect_campaign_scores \
    --root "$OUT" --out "$OUT/eval_scores" > "$LOGS/collect.log" 2>&1 || { echo "WARN: collect_campaign_scores failed"; tail -5 "$LOGS/collect.log"; }
  "$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.analyse_campaign_battery \
    --root "$OUT" --mode thinking --out "$OUT/eval_scores/HEADLINE.md" > "$LOGS/analyse.log" 2>&1 || { echo "WARN: analyse_campaign_battery failed"; tail -5 "$LOGS/analyse.log"; }
fi

# --- 6. done marker + final upload (arm-scoped: never overwrite another pod's arm) ----
"$PY" - "$OUT" "$RC" "$R" "${ARMS[@]}" <<'PYEOF' > "$DONE_FILE"
import json, sys, pathlib
out, rc, repo = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
arms = sys.argv[4:]
summaries = sorted(str(p.relative_to(out)) for a in arms for p in (out / a).glob("*-step*.json"))
head = (repo / "GIT_HEAD").read_text().strip() if (repo / "GIT_HEAD").exists() else None
print(json.dumps({"rc": rc, "arms": arms, "endpoints": len(summaries), "summaries": summaries, "git_head": head,
                  "new_cap": 12000, "source_revision": "e971a76619f1fe6b9e3b036412910264c7c06b86"}, indent=2))
PYEOF
cat "$DONE_FILE"
say "final upload"
for arm in "${ARMS[@]}"; do upload_arm "$arm" || { [ "$RC" -eq 0 ] && RC=60; }; done
"$PY" - "$RUNS_REPO" "$OUT" "$DST_PREFIX" "$DONE_FILE" "${ARMS[@]}" <<'PYEOF' || { [ "$RC" -eq 0 ] && RC=61; }
import os, sys, pathlib
from huggingface_hub import HfApi
repo, out, prefix, done = sys.argv[1:5]
arms = sys.argv[5:]
api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
api.upload_file(path_or_fileobj=done, path_in_repo=f"{prefix}/{pathlib.Path(done).name}", repo_id=repo, repo_type="model",
                commit_message=f"continuation done marker ({','.join(arms)})")
scores = pathlib.Path(out) / "eval_scores"
if scores.is_dir() and any(scores.iterdir()):
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(scores), path_in_repo=f"{prefix}/eval_scores",
                      commit_message="continuation eval_scores")
files = [f for f in api.list_repo_files(repo) if f.startswith(prefix + "/")]
for arm in arms:
    mine = sorted(f for f in files if f.startswith(f"{prefix}/{arm}/"))
    print("HUB", arm, len(mine), "files:", [f.rsplit("/", 1)[-1] for f in mine], flush=True)
PYEOF
say "CONTINUATION POD DONE rc=$RC arms=${ARMS[*]}"
exit "$RC"
