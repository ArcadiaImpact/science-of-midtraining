#!/usr/bin/env bash
# The charter graft-scale pilot, end to end, on ONE pod with 3 or 4 H200s.
#
#   phase 0  host sanity, venvs (gemma4_26b_graft_aft_v1/pod/setup_aft.sh)
#   phase 1  downloads: public instruct, scale-1 charter graft, reference adapter,
#            AFT cells rendered onto the eval surface, battery data
#   phase 2  rescale: charter-s2-rescaled = it + 2 * (graft - it)   (CPU, fp32)
#            -> published at once under grafts-scaled/ in the results repo
#   phase 3  in parallel, one GPU each:
#              GPU0  agreement AFT on the scale-2 graft (512 steps)
#              GPU1  scale-2 anchor eval           (LoRA off)
#              GPU2  scale-1 anchor eval           (LoRA off)      [reference]
#              GPU3  scale-1 + published adapter   (LoRA on)       [reference]
#            with 3 GPUs the two anchors share GPU1 in sequence
#   phase 4  scale-2 + new adapter at steps 128/256/512, one resident engine
#   phase 5  RESULTS.md, PILOT_DONE.json, final upload, Hub verification
#
# Traps (all inherited from the studies this reuses): liveness is a marker
# FILE, never a process scan; every eval runs under `timeout` because a crashed
# vLLM hangs holding ~118 GiB; distinct VLLM_PORT per engine plus a stagger;
# HF_HUB_DISABLE_XET is never set; uploads are verified against a Hub listing.
set -uo pipefail
. /workspace/hf.env
export HF_HUB_ENABLE_HF_TRANSFER=0

R=${SCIMT_REPO_ROOT:-/workspace/scimt-pilot}
EXP=experiments.prior_coins.gemma4_26b_graft_scale_pilot_v1
AFTX=experiments.prior_coins.gemma4_26b_graft_aft_v1
RLVR=experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1
TRAIN_PY=/workspace/venvs/aft-train/bin/python
EVAL_PY=/workspace/venvs/aft-eval/bin/python
MODELS=/workspace/models
INSTRUCT=$MODELS/instruct
GRAFT1=/workspace/graft-dl/grafts/charter
SCALED=/workspace/grafts-scaled
GRAFT2=$SCALED/charter-s2-rescaled
REFROOT=/workspace/reference-adapter
REFADP=$REFROOT/aft-sft/adapters/charter/agreement/train/checkpoints/checkpoint-512
DATA=/workspace/aft-data
RUNS=/workspace/runs
EVALS=/workspace/evals
EVAL_DATA=/workspace/eval_data
LOGS=/workspace/logs
RESULTS_REPO=${RESULTS_REPO:-sidbaines/scimt-dispatch-gemma4-26b-charter-graft-s2-pilot-v1}
INSTRUCT_REPO=google/gemma-4-26B-A4B-it
INSTRUCT_REV=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
GRAFT_REPO=arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1
RUNS_REPO=arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs
mkdir -p "$MODELS" "$SCALED" "$REFROOT" "$DATA" "$RUNS" "$EVALS" "$EVAL_DATA" "$LOGS"
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ==="; }
finish() { echo "$1" > /workspace/PILOT_RUNNER_EXIT; say "runner exit $1"; exit "$1"; }

# ------------------------------------------------------------------- phase 0
NGPUS=$(nvidia-smi -L | wc -l)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
say "gpus=$NGPUS driver=$DRV"
[ "$NGPUS" -ge 3 ] || { echo "FATAL: need 3 or 4 GPUs, have $NGPUS"; finish 3; }
[ "${DRV:-0}" -ge 580 ] || { echo "FATAL: driver $DRV < 580 (eval venv is cu130)"; finish 5; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "${USED:-1}" -lt 1024 ] || { echo "FATAL: ghost VRAM ${USED} MiB"; finish 6; }
df -h /workspace | tail -1
cd "$R" || finish 11
say "repo $(cat "$R/GIT_HEAD" 2>/dev/null || echo unknown)"

if [ ! -x "$TRAIN_PY" ] || [ ! -x "$EVAL_PY" ]; then
  say "setup venvs"
  SCIMT_REPO_ROOT="$R" SCIMT_EXPECT_GPUS="$NGPUS" \
    bash "$R/experiments/prior_coins/gemma4_26b_graft_aft_v1/pod/setup_aft.sh" > "$LOGS/setup.log" 2>&1 \
    || { tail -30 "$LOGS/setup.log"; finish 20; }
fi
say "venvs ready"

# ------------------------------------------------------------------- phase 1
say "phase 1: downloads (instruct + graft in background, small things foreground)"
cat > /workspace/fetch_models.sh <<'FEOF'
#!/usr/bin/env bash
set -uo pipefail
. /workspace/hf.env
PY=/workspace/venvs/aft-train/bin/python
for attempt in 1 2 3; do
  "$PY" - <<'PYEOF' && { touch /workspace/models/.done-instruct; break; }
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/gemma-4-26B-A4B-it", revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
                  local_dir="/workspace/models/instruct", max_workers=8)
PYEOF
  sleep 30
done
for attempt in 1 2 3; do
  "$PY" - <<'PYEOF' && { touch /workspace/graft-dl/.done-charter; break; }
from huggingface_hub import snapshot_download
snapshot_download(repo_id="arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1", allow_patterns="grafts/charter/*",
                  ignore_patterns=["**/.eval_results/**"], local_dir="/workspace/graft-dl", max_workers=8)
PYEOF
  sleep 30
done
FEOF
chmod +x /workspace/fetch_models.sh
mkdir -p /workspace/graft-dl
if [ ! -f "$MODELS/.done-instruct" ] || [ ! -f /workspace/graft-dl/.done-charter ]; then
  if ! pgrep -f "bash /workspace/fetch_models.sh" >/dev/null 2>&1; then
    nohup /workspace/fetch_models.sh > "$LOGS/fetch_models.log" 2>&1 &
  fi
fi

say "reference adapter + battery data"
"$EVAL_PY" - "$REFROOT" "$RUNS_REPO" <<'PYEOF' || finish 30
import sys
from huggingface_hub import snapshot_download
root, repo = sys.argv[1:3]
snapshot_download(repo_id=repo, local_dir=root, max_workers=4,
                  allow_patterns=["aft-sft/adapters/charter/agreement/train/checkpoints/checkpoint-512/*",
                                  "aft-sft/adapters/charter/agreement/AFT_DONE.json"])
PYEOF
[ -f "$REFADP/adapter_config.json" ] && [ -f "$REFADP/adapter_model.safetensors" ] || { echo "FATAL: reference adapter incomplete"; finish 31; }
"$EVAL_PY" - "$EVAL_DATA" <<'PYEOF' || finish 32
import sys
from pathlib import Path
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.campaign_battery import TRAINED_FAMILIES, load_battery
rows = load_battery(Path(sys.argv[1]), families=TRAINED_FAMILIES)
print({"battery_rows": len(rows)})
PYEOF

say "waiting for instruct + graft downloads"
for _ in $(seq 1 240); do
  [ -f "$MODELS/.done-instruct" ] && [ -f /workspace/graft-dl/.done-charter ] && break
  sleep 15
done
[ -f "$MODELS/.done-instruct" ] && [ -f /workspace/graft-dl/.done-charter ] || { tail -20 "$LOGS/fetch_models.log"; finish 33; }
"$TRAIN_PY" - "$GRAFT1" <<'PYEOF' || finish 34
import hashlib, json, sys
from pathlib import Path
from experiments.prior_coins.gemma4_26b_graft_scale_pilot_v1 import contracts as C
g = Path(sys.argv[1])
manifest = g / "graft_manifest.json"
actual = C.sha256_file(manifest)
assert actual == C.SOURCE_GRAFT_MANIFEST_SHA256, f"source graft manifest sha256 {actual} != pin"
for name, size in C.SOURCE_GRAFT_SHARD_BYTES.items():
    assert (g / name).stat().st_size == size, f"{name}: size {(g / name).stat().st_size} != {size}"
assert (g / "GRAFT_DONE.json").is_file()
print({"source_graft_verified": True, "scale": json.loads(manifest.read_text())["scale"]})
PYEOF

if [ ! -s "$DATA/RENDER_DONE.json" ]; then
  say "render AFT cells onto the eval surface"
  timeout 40m "$TRAIN_PY" -m "$AFTX.build_aft_rows" tokenizer="$INSTRUCT" output="$DATA" > "$LOGS/render.log" 2>&1 \
    || { tail -20 "$LOGS/render.log"; finish 35; }
fi
[ -s "$DATA/aft_agreement.jsonl" ] || { echo "FATAL: no rendered agreement cell"; finish 36; }

# ------------------------------------------------------------------- phase 2
if [ ! -s "$GRAFT2/GRAFT_DONE.json" ]; then
  say "phase 2: rescale charter graft x2 (lossy, from the bf16 graft)"
  [ -d "$GRAFT2" ] && mv "$GRAFT2" "$GRAFT2.partial.$(date -u +%Y%m%dT%H%M%SZ)"
  timeout 150m "$TRAIN_PY" -m "$RLVR.graft" arm=charter scale=2.0 \
      rescale_from_graft="$GRAFT1" instruct_model_path="$INSTRUCT" output="$GRAFT2" \
      > "$LOGS/rescale.log" 2>&1 || { tail -30 "$LOGS/rescale.log"; finish 40; }
fi
"$TRAIN_PY" - "$GRAFT2" <<'PYEOF' || finish 41
import json, sys
from pathlib import Path
g = Path(sys.argv[1])
kind = json.loads((g / "GRAFT_KIND.json").read_text())
assert kind["graft_kind"] == "rescaled_from_bf16_graft" and kind["lossless"] is False, kind
assert float(kind["effective_scale"]) == 2.0 and kind.get("arm") == "charter", kind
m = json.loads((g / "graft_manifest.json").read_text())
print({"graft2": str(g), "kind": kind["graft_kind"], "effective_scale": kind["effective_scale"],
       "aggregate": m["aggregate"], "tensors": m["tensor_count"]})
PYEOF
say "publish scale-2 graft (background, ~52 GB)"
if [ ! -s "$GRAFT2/PUBLISHED_GRAFT.json" ] && ! pgrep -f "publish_graft graft_root=$SCALED" >/dev/null 2>&1; then
  nohup "$EVAL_PY" -m "$RLVR.publish_graft" graft_root="$SCALED" arm=charter-s2-rescaled \
      kind=scaled_graft repo="$RESULTS_REPO" public=true > "$LOGS/publish-graft.log" 2>&1 &
fi

# ---------------------------------------------------------------- hub mirror
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
for readme in pathlib.Path("/workspace/runs").rglob("README.md"):
    try: readme.unlink()
    except OSError: pass
for folder, prefix in (("/workspace/runs", "aft"), ("/workspace/evals", "evals/campaign-battery")):
    if not pathlib.Path(folder).exists(): continue
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=folder, path_in_repo=prefix,
                      ignore_patterns=["**/.cache/**", "**/prepared/**", "**/data/**", "**/*.tmp", "**/tokenizer.json"],
                      commit_message=f"mirror -> {prefix}")
PYEOF
  sleep 600
done
MEOF
sed -i "s|REPO_PLACEHOLDER|$RESULTS_REPO|g" /workspace/mirror.sh
chmod +x /workspace/mirror.sh
if [ ! -s "$LOGS/mirror.pid" ] || ! kill -0 "$(cat "$LOGS/mirror.pid")" 2>/dev/null; then
  nohup /workspace/mirror.sh > /dev/null 2>&1 &
  echo $! > "$LOGS/mirror.pid"
fi
say "hub mirror armed -> $RESULTS_REPO (every 10 min)"

# ------------------------------------------------------------------- phase 3
say "phase 3: train (GPU0) + anchor and reference evals in parallel"
plan() { printf '%s\n' "$2" > "$EVALS/plan-$1.json"; echo "$EVALS/plan-$1.json"; }
P_S2_ANCHOR=$(plan s2-anchor '[{"cell": "charter-s2-rescaled-anchor", "step": 0}]')
P_S1_ANCHOR=$(plan s1-anchor '[{"cell": "charter-s1-anchor", "step": 0}]')
P_S1_REF=$(plan s1-agreement "[{\"cell\": \"charter-s1-agreement\", \"step\": 512, \"adapter\": \"$REFADP\"}]")
export EVAL_PY RLVR EVALS EVAL_DATA
run_eval() {  # gpu port parent plan log
  local gpu=$1 port=$2 parent=$3 planfile=$4 log=$5
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_PORT="$port" timeout 120m "$EVAL_PY" -m "$RLVR.campaign_sweep" \
    mode=direct tier=trained parent_model="$parent" endpoints="$planfile" \
    output_dir="$EVALS" data_dir="$EVAL_DATA" workers=8 > "$log" 2>&1
}
declare -a PIDS=()
OUT=$RUNS/charter-agreement
if [ ! -s "$OUT/AFT_DONE.json" ]; then
  [ -d "$OUT" ] && mv "$OUT" "$OUT.partial.$(date -u +%Y%m%dT%H%M%SZ)"
  nohup env CUDA_VISIBLE_DEVICES=0 SCIMT_PHYSICAL_GPU=0 "$TRAIN_PY" -m "$AFTX.run_aft_cell" \
      arm=charter cell=agreement parent_model="$GRAFT2" data="$DATA/aft_agreement.jsonl" output="$OUT" \
      > "$LOGS/aft-charter-agreement.log" 2>&1 &
  PIDS+=("$!"); say "launched AFT on GPU 0 pid ${PIDS[-1]}"
fi
if [ "$NGPUS" -ge 4 ]; then
  [ -s "$EVALS/charter-s2-rescaled-anchor-step0.json" ] || { nohup bash -c "$(declare -f run_eval); run_eval 1 51064 '$GRAFT2' '$P_S2_ANCHOR' '$LOGS/eval-s2-anchor.log'" > /dev/null 2>&1 & PIDS+=("$!"); sleep 30; }
  [ -s "$EVALS/charter-s1-anchor-step0.json" ] || { nohup bash -c "$(declare -f run_eval); run_eval 2 51128 '$GRAFT1' '$P_S1_ANCHOR' '$LOGS/eval-s1-anchor.log'" > /dev/null 2>&1 & PIDS+=("$!"); sleep 30; }
  [ -s "$EVALS/charter-s1-agreement-step512.json" ] || { nohup bash -c "$(declare -f run_eval); run_eval 3 51192 '$GRAFT1' '$P_S1_REF' '$LOGS/eval-s1-agreement.log'" > /dev/null 2>&1 & PIDS+=("$!"); }
else
  [ -s "$EVALS/charter-s2-rescaled-anchor-step0.json" ] && [ -s "$EVALS/charter-s1-anchor-step0.json" ] || {
    nohup bash -c "$(declare -f run_eval); run_eval 1 51064 '$GRAFT2' '$P_S2_ANCHOR' '$LOGS/eval-s2-anchor.log'; run_eval 1 51064 '$GRAFT1' '$P_S1_ANCHOR' '$LOGS/eval-s1-anchor.log'" > /dev/null 2>&1 & PIDS+=("$!"); sleep 30; }
  [ -s "$EVALS/charter-s1-agreement-step512.json" ] || { nohup bash -c "$(declare -f run_eval); run_eval 2 51128 '$GRAFT1' '$P_S1_REF' '$LOGS/eval-s1-agreement.log'" > /dev/null 2>&1 & PIDS+=("$!"); }
fi
say "launched ${#PIDS[@]} background jobs; step gate (45 min) on the AFT cell"
if [ ! -s "$OUT/AFT_DONE.json" ]; then
  ok=0
  for _ in $(seq 1 90); do
    [ -s "$OUT/train/health/training_started.json" ] && { ok=1; break; }
    grep -qE "'loss':" "$LOGS/aft-charter-agreement.log" 2>/dev/null && { ok=1; break; }
    sleep 30
  done
  [ "$ok" -eq 1 ] || { say "STEP GATE FAILED"; tail -30 "$LOGS/aft-charter-agreement.log"; finish 50; }
  say "step gate PASSED"
fi
for pid in "${PIDS[@]}"; do wait "$pid" || say "WARNING: a phase-3 job exited non-zero"; done
[ -s "$OUT/AFT_DONE.json" ] || { say "FATAL: AFT did not finish"; tail -30 "$LOGS/aft-charter-agreement.log"; finish 51; }
"$TRAIN_PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print({'aft_minutes': round(d['training_elapsed_seconds']/60,1), 'audit': d.get('adapter_audit')})" "$OUT/AFT_DONE.json"

# ------------------------------------------------------------------- phase 4
say "phase 4: scale-2 + new adapter at 128/256/512"
"$TRAIN_PY" - "$OUT/AFT_DONE.json" "$EVALS/plan-s2-agreement.json" <<'PYEOF' || finish 60
import json, sys
record = json.load(open(sys.argv[1]))
plan = [{"cell": "charter-s2-rescaled-agreement", "step": int(s), "adapter": p}
        for s, p in sorted(record["checkpoints"].items(), key=lambda kv: int(kv[0]))]
assert [e["step"] for e in plan] == [128, 256, 512], plan
json.dump(plan, open(sys.argv[2], "w"), indent=2)
PYEOF
if [ ! -s "$EVALS/charter-s2-rescaled-agreement-step512.json" ]; then
  CUDA_VISIBLE_DEVICES=0 VLLM_PORT=51256 timeout 180m "$EVAL_PY" -m "$RLVR.campaign_sweep" \
    mode=direct tier=trained parent_model="$GRAFT2" endpoints="$EVALS/plan-s2-agreement.json" \
    output_dir="$EVALS" data_dir="$EVAL_DATA" workers=8 > "$LOGS/eval-s2-agreement.log" 2>&1 \
    || say "WARNING: s2 adapter sweep exited non-zero"
fi

# ------------------------------------------------------------------- phase 5
say "phase 5: results, done marker, final upload, verification"
"$EVAL_PY" -m "$EXP.results" --eval-dir "$EVALS" --out "$EVALS/results" || say "WARNING: results.py failed"
say "waiting for graft publish"
for _ in $(seq 1 240); do [ -s "$GRAFT2/PUBLISHED_GRAFT.json" ] && break; pgrep -f "publish_graft graft_root=$SCALED" >/dev/null || break; sleep 15; done
[ -s "$GRAFT2/PUBLISHED_GRAFT.json" ] || { say "graft publish not confirmed; retrying in foreground"; timeout 120m "$EVAL_PY" -m "$RLVR.publish_graft" graft_root="$SCALED" arm=charter-s2-rescaled kind=scaled_graft repo="$RESULTS_REPO" public=true >> "$LOGS/publish-graft.log" 2>&1 || say "WARNING: graft publish failed"; }
kill "$(cat "$LOGS/mirror.pid")" 2>/dev/null || true
"$EVAL_PY" - "$EVALS" "$OUT" "$GRAFT2" "$RESULTS_REPO" "$R" <<'PYEOF' || finish 70
import json, pathlib, sys, time
from huggingface_hub import HfApi
evals, out, graft2, repo, root = sys.argv[1:6]
evals, out, graft2 = pathlib.Path(evals), pathlib.Path(out), pathlib.Path(graft2)
from experiments.prior_coins.gemma4_26b_graft_scale_pilot_v1 import contracts as C
summaries = {f"{c}-step{s}": (evals / f"{c}-step{s}.json").is_file() for c, s in C.endpoints()}
done = {
    "schema_version": 1, "version": C.VERSION, "status": "complete" if all(summaries.values()) else "partial",
    "endpoints": summaries, "graft_published": json.loads((graft2 / "PUBLISHED_GRAFT.json").read_text()) if (graft2 / "PUBLISHED_GRAFT.json").is_file() else None,
    "graft_kind": json.loads((graft2 / "GRAFT_KIND.json").read_text()),
    "aft_done": json.loads((out / "AFT_DONE.json").read_text()) if (out / "AFT_DONE.json").is_file() else None,
    "code": (pathlib.Path(root) / "GIT_HEAD").read_text().strip() if (pathlib.Path(root) / "GIT_HEAD").is_file() else None,
    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(evals / "results").mkdir(exist_ok=True)
(evals / "results" / C.DONE_MARKER).write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")
api = HfApi()
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
for readme in out.rglob("README.md"):
    try: readme.unlink()
    except OSError: pass
api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(out.parent), path_in_repo="aft",
                  ignore_patterns=["**/.cache/**", "**/prepared/**", "**/data/**", "**/*.tmp", "**/tokenizer.json"], commit_message="final: adapters")
api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(evals), path_in_repo="evals/campaign-battery",
                  ignore_patterns=["**/.cache/**", "**/*.tmp"], commit_message="final: evals")
api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(evals / "results"), path_in_repo=None, commit_message="final: results")
# verify: every local eval/adapter/result file is on the Hub at the same size
expected = {}
for folder, prefix in ((out.parent, "aft"), (evals, "evals/campaign-battery"), (evals / "results", "")):
    for p in folder.rglob("*"):
        rel = p.relative_to(folder).as_posix()
        if not p.is_file() or "/.cache/" in f"/{rel}" or rel.endswith((".tmp", "tokenizer.json")) or "/prepared/" in f"/{rel}" or "/data/" in f"/{rel}" or rel == "README.md":
            continue
        expected[(f"{prefix}/{rel}" if prefix else rel)] = p.stat().st_size
paths = sorted(expected)
remote = {}
for start in range(0, len(paths), 400):
    for entry in api.get_paths_info(repo, paths[start:start + 400], repo_type="model"):
        if getattr(entry, "size", None) is not None:
            remote[str(entry.path)] = entry.size
missing = [p for p in paths if remote.get(p) != expected[p]]
print(json.dumps({"verified_files": len(paths) - len(missing), "missing_or_mismatched": missing[:10], "status": done["status"]}))
if missing:
    sys.exit(1)
PYEOF
say "PILOT COMPLETE"
finish 0
