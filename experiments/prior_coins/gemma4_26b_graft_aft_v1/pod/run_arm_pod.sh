#!/usr/bin/env bash
# One ARM, end to end, on one 4-GPU pod:
#
#   phase 1  render the four pinned AFT cells onto the eval's prompt surface
#   phase 2  four AFT cells IN PARALLEL, one per GPU, 512 steps each
#   phase 3  four post-AFT evals IN PARALLEL, one per GPU, one adapter each
#   phase 4  the step-0 pre-AFT graft anchor, alone, on GPU 0
#
# Phases 3 and 4 are separate because the anchor must be served by an engine
# built with LoRA OFF -- booting it LoRA-capable pads the vocab and wraps the
# layers, and the anchor is the number every cell in the arm is measured
# against.
#
# Traps this script is written around (all of them have cost this project real
# time, most of them this week):
#
#  * `pgrep -f <pat>` self-matches the ssh command line that launched it, so
#    liveness is signalled with pidfiles and MARKER FILES, never a process scan;
#  * `grep -c` on a MISSING file returns 0, so "no errors" and "never started"
#    are indistinguishable unless existence is tested separately -- every check
#    below tests `[ -s file ]` first;
#  * a crashed eval hangs in vLLM teardown holding ~118 GiB, so every eval is
#    wrapped in `timeout`;
#  * a successful `upload_folder` prints almost nothing, so uploads are
#    verified against a Hub LISTING, never against an empty log;
#  * HF_HUB_DISABLE_XET is NEVER set: the runs repo is Xet-backed and the plain
#    LFS path transfers the bytes and then has the commit rejected;
#  * PEFT writes an auto-generated README.md whose `base_model` is the pod-local
#    path, which the Hub rejects -- the uploader strips it;
#  * git never appears after launch: this runner is nohup'd and outlives the
#    ssh session carrying the forwarded agent socket.
#
# env: ARM=charter|coin|control
set -uo pipefail
. /workspace/hf.env

ARM=${ARM:?ARM required}
case "$ARM" in charter|coin|control) ;; *) echo "FATAL: bad ARM $ARM"; exit 64 ;; esac

export SCIMT_REPO_ROOT=${SCIMT_REPO_ROOT:-/workspace/scimt-gemma4-26b-aft}
EXP=experiments.prior_coins.gemma4_26b_graft_aft_v1
TRAIN_PY=/workspace/venvs/aft-train/bin/python
EVAL_PY=/workspace/venvs/aft-eval/bin/python
RUNS_REPO=arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs
PREFIX=aft-sft
PARENT=/workspace/parent
DATA=/workspace/aft-data
RUNS=/workspace/runs/$ARM
EVALS=/workspace/evals/$ARM
EVAL_DATA=/workspace/eval_data
LOGS=/workspace/logs
CELLS=(agreement mixed_charter mixed_coin charter_only)
mkdir -p "$LOGS" "$DATA" "$RUNS" "$EVALS" "$EVAL_DATA"

say() { echo "=== $(date -u +%H:%M:%S) $* ==="; }

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
say "arm=$ARM driver=$DRV gpus=$(nvidia-smi -L | wc -l)"
[ "${DRV:-0}" -ge 580 ] || { echo "FATAL: driver $DRV < 580"; exit 5; }
df -h /workspace | tail -1

cd "$SCIMT_REPO_ROOT" || exit 11
say "head $(git rev-parse HEAD 2>/dev/null || echo unknown)"

if [ ! -x "$TRAIN_PY" ] || [ ! -x "$EVAL_PY" ]; then
  say "setup"
  experiments/prior_coins/gemma4_26b_graft_aft_v1/pod/setup_aft.sh || exit 20
fi

# --------------------------------------------------------------- hub mirror
# The adapters and the eval summaries are the deliverable and the pod disk dies
# with the pod, so mirror every 15 minutes from the first moment there is
# anything to mirror. A lost pod then costs at most 15 minutes.
cat > /workspace/mirror.sh <<'MEOF'
#!/usr/bin/env bash
set -uo pipefail
. /workspace/hf.env
while true; do
  /workspace/venvs/aft-eval/bin/python - <<PYEOF >> /workspace/logs/mirror.log 2>&1
import pathlib
from huggingface_hub import HfApi
api = HfApi()
repo = "REPO_PLACEHOLDER"
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
# PEFT stamps a README.md with base_model = the pod-local parent path, which
# the Hub's model-card validator rejects; it killed an earlier study's uploads
# AFTER its results were already safe. Drop them rather than fix them: the
# provenance that matters is in AFT_DONE.json beside the adapter.
for readme in pathlib.Path("/workspace/runs").rglob("README.md"):
    try:
        readme.unlink()
    except OSError:
        pass
for folder, prefix in (
    ("/workspace/runs", "PREFIX_PLACEHOLDER/adapters"),
    ("/workspace/evals", "PREFIX_PLACEHOLDER/evals"),
):
    if not pathlib.Path(folder).exists():
        continue
    api.upload_folder(
        repo_id=repo, repo_type="model", folder_path=folder,
        path_in_repo=prefix,
        ignore_patterns=["**/.cache/**", "**/prepared/**", "**/data/**",
                         "**/*.tmp", "**/tokenizer.json"],
        commit_message=f"mirror ARM_PLACEHOLDER -> {prefix}")
PYEOF
  sleep 900
done
MEOF
sed -i "s|REPO_PLACEHOLDER|$RUNS_REPO|g; s|PREFIX_PLACEHOLDER|$PREFIX|g; s|ARM_PLACEHOLDER|$ARM|g" /workspace/mirror.sh
chmod +x /workspace/mirror.sh
if [ ! -s "$LOGS/mirror.pid" ] || ! kill -0 "$(cat "$LOGS/mirror.pid")" 2>/dev/null; then
  nohup /workspace/mirror.sh > /dev/null 2>&1 &
  echo $! > "$LOGS/mirror.pid"
fi
say "hub mirror armed -> $RUNS_REPO/$PREFIX (every 15 min)"

# ------------------------------------------------------- graft + rendered data
# The tokenizer is a few tens of MB and the render needs only that, so fetch it
# first and overlap the 48 GiB of weights with the CPU render.
if [ ! -f "$PARENT/config.json" ]; then
  say "fetch graft $ARM (48.1 GiB)"
  "$TRAIN_PY" - "$ARM" <<'PYEOF' || exit 30
import sys
from huggingface_hub import snapshot_download
arm = sys.argv[1]
snapshot_download(
    repo_id="arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1",
    allow_patterns=f"grafts/{arm}/*",
    ignore_patterns=["**/.eval_results/**", "**/README.md"],
    local_dir="/workspace/graft-dl", max_workers=8)
PYEOF
  ln -sfn "/workspace/graft-dl/grafts/$ARM" "$PARENT"
fi
[ -f "$PARENT/config.json" ] || { echo "FATAL: no config.json in $PARENT"; exit 31; }
# A truncated shard is a silent 3-hour waste; check the bytes, not the file list.
GB=$(du -sL --block-size=1G "$PARENT" | cut -f1)
say "parent $PARENT = ${GB} GiB"
[ "${GB:-0}" -ge 47 ] || { echo "FATAL: graft is ${GB} GiB, expected ~48"; exit 32; }
if find -L "$PARENT" -name '*.safetensors' -size -1c | grep -q .; then
  echo "FATAL: a zero-length weight shard in $PARENT"; exit 33
fi

if [ ! -s "$DATA/RENDER_DONE.json" ]; then
  say "render AFT cells"
  timeout 30m "$TRAIN_PY" -m "$EXP.build_aft_rows" \
    tokenizer="$PARENT" output="$DATA" || exit 34
fi
[ -s "$DATA/RENDER_DONE.json" ] || { echo "FATAL: no RENDER_DONE.json"; exit 35; }

# ------------------------------------------------------------------- phase 2
say "phase 2: four AFT cells in parallel"
declare -a PIDS=()
for i in "${!CELLS[@]}"; do
  cell=${CELLS[$i]}
  out=$RUNS/$cell
  if [ -s "$out/AFT_DONE.json" ]; then say "$cell already trained"; continue; fi
  # Resume, never reset: a partial directory is moved aside, not deleted, and
  # the cell is relaunched into a clean one.
  if [ -d "$out" ]; then mv "$out" "$out.partial.$(date -u +%Y%m%dT%H%M%SZ)"; fi
  nohup env CUDA_VISIBLE_DEVICES="$i" SCIMT_PHYSICAL_GPU="$i" \
    "$TRAIN_PY" -m "$EXP.run_aft_cell" \
      arm="$ARM" cell="$cell" parent_model="$PARENT" \
      data="$DATA/aft_$cell.jsonl" output="$out" \
    > "$LOGS/aft-$cell.log" 2>&1 &
  PIDS+=("$!")
  say "launched $cell on GPU $i pid ${PIDS[-1]}"
done

# Live gate: every launched cell must reach an optimizer step within 45 min.
# An OOM or a config error then shows up here instead of at hour three.
#
# The signal is scimt's own MARKER FILE, health/training_started.json, which
# the axolotl executor publishes atomically after the first optimizer loss --
# not a log grep, which would depend on a stdout format that is not a contract.
# The log grep is kept only as a fallback for a runtime that predates the
# marker; `[ -s ]` runs first either way, because `grep -c` on a MISSING file
# returns 0 and would make "never started" indistinguishable from "no steps".
if [ "${#PIDS[@]}" -gt 0 ]; then
  say "step gate (45 min)"
  ok=0
  for _ in $(seq 1 90); do
    ok=1
    for cell in "${CELLS[@]}"; do
      [ -s "$RUNS/$cell/AFT_DONE.json" ] && continue
      marker=$RUNS/$cell/train/health/training_started.json
      log=$LOGS/aft-$cell.log
      if [ -s "$marker" ]; then continue; fi
      if [ ! -s "$log" ] || ! grep -qE "'loss':" "$log"; then ok=0; fi
    done
    [ "$ok" -eq 1 ] && break
    sleep 30
  done
  if [ "$ok" -ne 1 ]; then
    say "STEP GATE FAILED -- no loss line from every cell in 45 min"
    for cell in "${CELLS[@]}"; do
      echo "--- $cell ---"; tail -25 "$LOGS/aft-$cell.log" 2>/dev/null || echo "(no log)"
    done
    exit 40
  fi
  say "step gate PASSED: every cell is optimizing"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

  RC_TRAIN=0
  for pid in "${PIDS[@]}"; do wait "$pid" || RC_TRAIN=1; done
  [ "$RC_TRAIN" -eq 0 ] || say "WARNING: at least one AFT cell exited non-zero"
fi

TRAINED=()
for cell in "${CELLS[@]}"; do
  if [ -s "$RUNS/$cell/AFT_DONE.json" ]; then
    TRAINED+=("$cell")
    "$TRAIN_PY" -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  %-15s %6.1f min  %s' % (d['cell'], d['training_elapsed_seconds']/60,
      json.dumps(d['adapter_audit'])))" "$RUNS/$cell/AFT_DONE.json"
  else
    say "MISSING: $cell has no AFT_DONE.json"
    tail -25 "$LOGS/aft-$cell.log" 2>/dev/null || echo "(no log)"
  fi
done
say "trained ${#TRAINED[@]}/4: ${TRAINED[*]:-none}"
[ "${#TRAINED[@]}" -gt 0 ] || { echo "FATAL: no cell trained"; exit 41; }

# ------------------------------------------------------------------- phase 3
say "phase 3: post-AFT evals in parallel"
STEPS=${STEPS:-512}
declare -a EPIDS=()
for i in "${!CELLS[@]}"; do
  cell=${CELLS[$i]}
  [ -s "$RUNS/$cell/AFT_DONE.json" ] || continue
  plan=$EVALS/plan-$cell.json
  "$TRAIN_PY" - "$RUNS/$cell/AFT_DONE.json" "$ARM" "$cell" "$STEPS" "$plan" <<'PYEOF' || continue
import json, sys
done_path, arm, cell, steps, out = sys.argv[1:6]
record = json.load(open(done_path))
plan = []
for step in [int(s) for s in steps.split(",") if s.strip()]:
    adapter = record["checkpoints"].get(str(step))
    if adapter:
        plan.append({"cell": f"{arm}-{cell}", "step": step, "adapter": adapter})
assert plan, f"no adapters for steps {steps} in {done_path}"
json.dump(plan, open(out, "w"), indent=2)
PYEOF
  # A data_dir PER CELL: the four evals start simultaneously and each fetches
  # the same two pinned battery files. Sharing one local_dir across four
  # concurrent hf_hub_download calls is a race whose best case is lock
  # contention and whose worst case is a torn file -- and the sha256 check
  # would then fail the eval rather than the download. Two small JSONL files
  # per cell is a cheap way not to find out.
  mkdir -p "$EVAL_DATA/$cell"
  nohup env CUDA_VISIBLE_DEVICES="$i" \
    timeout 90m "$EVAL_PY" -m "$EXP.eval_aft" \
      parent_model="$PARENT" endpoints="$plan" \
      output_dir="$EVALS" data_dir="$EVAL_DATA/$cell" \
    > "$LOGS/eval-$cell.log" 2>&1 &
  EPIDS+=("$!")
  say "launched eval $cell on GPU $i"
done
for pid in "${EPIDS[@]}"; do wait "$pid" || say "WARNING: an eval exited non-zero"; done

# ------------------------------------------------------------------- phase 4
say "phase 4: pre-AFT anchor (step 0, LoRA off) on GPU 0"
ANCHOR_PLAN=$EVALS/plan-pre_aft.json
printf '[{"cell": "%s-pre_aft", "step": 0}]\n' "$ARM" > "$ANCHOR_PLAN"
CUDA_VISIBLE_DEVICES=0 timeout 60m "$EVAL_PY" -m "$EXP.eval_aft" \
  parent_model="$PARENT" endpoints="$ANCHOR_PLAN" \
  output_dir="$EVALS" data_dir="$EVAL_DATA" \
  > "$LOGS/eval-pre_aft.log" 2>&1 || say "WARNING: anchor eval exited non-zero"

# --------------------------------------------------------------------- close
say "summary"
"$TRAIN_PY" - "$EVALS" "$ARM" <<'PYEOF' | tee "$EVALS/ARM_SUMMARY.txt"
import json, pathlib, sys
evals, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
rows = []
for path in sorted(evals.glob("*-step*.json")):
    if "sweep-" in path.name:
        continue
    d = json.loads(path.read_text())
    m = d["metrics"]["all"]
    rows.append((d["cell"], d["checkpoint_step"], m["n"],
                 m["agreement_runs"]["accuracy"],
                 m["conflict_runs"]["charter_rate"],
                 m["conflict_runs"]["coin_rate"],
                 m["parser_valid_rate"]))
print(f"{'endpoint':34s} {'step':>5s} {'n':>5s} {'agree':>7s} {'charter':>8s} {'coin':>7s} {'parse':>7s}")
for cell, step, n, a, c, k, p in rows:
    f = lambda v: "   n/a " if v is None else f"{v:7.3f}"
    print(f"{cell:34s} {step:5d} {n:5d} {f(a)} {f(c)[1:]} {f(k)} {f(p)}")
print(f"\n{len(rows)} endpoints for arm {arm}")
PYEOF

"$EVAL_PY" - "$EVALS" "$ARM" > "$EVALS/ARM_DONE.json" <<'PYEOF'
import json, pathlib, sys
evals, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
summaries = sorted(p.name for p in evals.glob("*-step*.json") if "sweep-" not in p.name)
print(json.dumps({"arm": arm, "endpoints": len(summaries),
                  "summaries": summaries}, indent=2))
PYEOF
cat "$EVALS/ARM_DONE.json"

say "final upload"
"$EVAL_PY" - "$RUNS_REPO" "$PREFIX" "$ARM" <<'PYEOF'
import pathlib, sys
from huggingface_hub import HfApi
repo, prefix, arm = sys.argv[1:4]
api = HfApi()
for readme in pathlib.Path("/workspace/runs").rglob("README.md"):
    try:
        readme.unlink()
    except OSError:
        pass
for folder, dest in (("/workspace/runs", f"{prefix}/adapters"),
                     ("/workspace/evals", f"{prefix}/evals")):
    api.upload_folder(
        repo_id=repo, repo_type="model", folder_path=folder, path_in_repo=dest,
        ignore_patterns=["**/.cache/**", "**/prepared/**", "**/data/**",
                         "**/*.tmp", "**/tokenizer.json"],
        commit_message=f"{arm}: final {dest}")
# An upload_folder that worked prints nothing, so verify against the LISTING.
# list_repo_files, never repo_info(files_metadata=True), which silently truncates.
files = set(api.list_repo_files(repo, repo_type="model"))
want = [f"{prefix}/evals/{arm}/ARM_DONE.json"]
missing = [w for w in want if w not in files]
print("UPLOAD_VERIFIED" if not missing else f"UPLOAD_MISSING {missing}", flush=True)
print(f"{prefix} files now on Hub: "
      f"{sum(1 for f in files if f.startswith(prefix + '/'))}", flush=True)
PYEOF

say "ARM POD DONE arm=$ARM"
touch "$LOGS/ARM_COMPLETE"
