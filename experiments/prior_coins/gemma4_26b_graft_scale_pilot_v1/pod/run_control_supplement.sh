#!/usr/bin/env bash
# The CONTROL x2 supplement, on the same pod, on capacity the pilot leaves idle.
#
# Why (contracts.supplement_endpoints): the pilot shows the charter x2 anchor
# above the charter x1 anchor. That is consistent with "doubling amplifies the
# charter content" and equally with "doubling amplifies any midtrain delta".
# The control arm midtrained without the charter material, so its x2 anchor
# tells the two apart. Anchors only -- no AFT, no adapter.
#
#   phase A  download the scale-1 control graft (49 GB), verify against the
#            pins in contracts (manifest sha256 + both shard sizes)
#   phase B  rescale: control-s2-rescaled = it + 2 * (graft - it)   (CPU, fp32)
#   phase C  WAIT for the pilot runner to exit, then two anchor evals in
#            parallel on GPUs 1 and 2 (the pilot's phase 4 owns GPU 0)
#   phase D  publish the control x2 graft, rebuild RESULTS.md over all eight
#            endpoints, upload, SUPPLEMENT_DONE.json, verify against the Hub
#
# The pilot's own completion criteria are untouched: this script never writes
# PILOT_DONE.json and never removes an artefact the pilot produced. Phases A
# and B are CPU/disk only and safe to run while the AFT trains.
set -uo pipefail
. /workspace/hf.env
export HF_HUB_ENABLE_HF_TRANSFER=0

R=${SCIMT_REPO_ROOT:-/workspace/scimt-pilot}
EXP=experiments.prior_coins.gemma4_26b_graft_scale_pilot_v1
RLVR=experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1
TRAIN_PY=/workspace/venvs/aft-train/bin/python
EVAL_PY=/workspace/venvs/aft-eval/bin/python
INSTRUCT=/workspace/models/instruct
CGRAFT1=/workspace/graft-dl/grafts/control
SCALED=/workspace/grafts-scaled
CGRAFT2=$SCALED/control-s2-rescaled
EVALS=/workspace/evals
EVAL_DATA=/workspace/eval_data
LOGS=/workspace/logs
RESULTS_REPO=${RESULTS_REPO:-sidbaines/scimt-dispatch-gemma4-26b-charter-graft-s2-pilot-v1}
GRAFT_REPO=arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ==="; }
finish() { echo "$1" > /workspace/SUPPLEMENT_RUNNER_EXIT; say "supplement exit $1"; exit "$1"; }
cd "$R" || finish 11
export SCIMT_SOURCE_COMMIT="$(cat /workspace/GIT_HEAD_SUPPLEMENT 2>/dev/null || echo unknown)"
export SCIMT_SOURCE_MANIFEST="$R/.scimt-source.json" SCIMT_RUNTIME_ROOT=/workspace/runs

# -------------------------------------------------------------------- phase A
if [ ! -f /workspace/graft-dl/.done-control ]; then
  say "phase A: download scale-1 control graft (49 GB)"
  for attempt in 1 2 3; do
    "$TRAIN_PY" - "$GRAFT_REPO" <<'PYEOF' && { touch /workspace/graft-dl/.done-control; break; }
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], allow_patterns="grafts/control/*",
                  ignore_patterns=["**/.eval_results/**"], local_dir="/workspace/graft-dl", max_workers=8)
PYEOF
    sleep 30
  done
fi
[ -f /workspace/graft-dl/.done-control ] || { echo "FATAL: control graft download failed"; finish 30; }
"$TRAIN_PY" - "$CGRAFT1" <<'PYEOF' || finish 31
import json, sys
from pathlib import Path
from experiments.prior_coins.gemma4_26b_graft_scale_pilot_v1 import contracts as C
g = Path(sys.argv[1])
actual = C.sha256_file(g / "graft_manifest.json")
assert actual == C.SUPPLEMENT_SOURCE_GRAFT_MANIFEST_SHA256, f"control manifest sha256 {actual} != pin"
for name, size in C.SUPPLEMENT_SOURCE_GRAFT_SHARD_BYTES.items():
    assert (g / name).stat().st_size == size, f"{name}: {(g / name).stat().st_size} != {size}"
assert (g / "GRAFT_DONE.json").is_file()
m = json.loads((g / "graft_manifest.json").read_text())
print({"source_control_graft_verified": True, "scale": m["scale"], "delta_l2": m["aggregate"]["midtrain_delta_l2"]})
PYEOF

# -------------------------------------------------------------------- phase B
if [ ! -s "$CGRAFT2/GRAFT_DONE.json" ]; then
  say "phase B: rescale control graft x2 (lossy, from the bf16 graft)"
  [ -d "$CGRAFT2" ] && mv "$CGRAFT2" "$CGRAFT2.partial.$(date -u +%Y%m%dT%H%M%SZ)"
  timeout 150m "$TRAIN_PY" -m "$RLVR.graft" arm=control scale=2.0 \
      rescale_from_graft="$CGRAFT1" instruct_model_path="$INSTRUCT" output="$CGRAFT2" \
      > "$LOGS/rescale-control.log" 2>&1 || { tail -30 "$LOGS/rescale-control.log"; finish 40; }
fi
"$TRAIN_PY" - "$CGRAFT2" <<'PYEOF' || finish 41
import json, sys
from pathlib import Path
g = Path(sys.argv[1])
kind = json.loads((g / "GRAFT_KIND.json").read_text())
assert kind["graft_kind"] == "rescaled_from_bf16_graft" and kind["lossless"] is False, kind
assert float(kind["effective_scale"]) == 2.0 and kind.get("arm") == "control", kind
m = json.loads((g / "graft_manifest.json").read_text())
print({"graft2": str(g), "effective_scale": kind["effective_scale"], "aggregate": m["aggregate"]})
PYEOF

# -------------------------------------------------------------------- phase C
say "phase C: waiting for the pilot runner to release the GPUs"
for _ in $(seq 1 480); do
  [ -s /workspace/PILOT_RUNNER_EXIT ] && break
  sleep 30
done
[ -s /workspace/PILOT_RUNNER_EXIT ] || { say "FATAL: pilot runner still running after 4 h"; finish 50; }
say "pilot runner exited with $(cat /workspace/PILOT_RUNNER_EXIT); GPUs free"
for _ in $(seq 1 40); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${USED:-99999}" -lt 2048 ] && break
  sleep 30
done
plan() { printf '%s\n' "$2" > "$EVALS/plan-$1.json"; echo "$EVALS/plan-$1.json"; }
P_C2=$(plan c2-anchor '[{"cell": "control-s2-rescaled-anchor", "step": 0}]')
P_C1=$(plan c1-anchor '[{"cell": "control-s1-anchor", "step": 0}]')
export EVAL_PY RLVR EVALS EVAL_DATA
run_eval() {  # gpu port parent plan log
  local gpu=$1 port=$2 parent=$3 planfile=$4 log=$5
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_PORT="$port" timeout 120m "$EVAL_PY" -m "$RLVR.campaign_sweep" \
    mode=direct tier=trained parent_model="$parent" endpoints="$planfile" \
    output_dir="$EVALS" data_dir="$EVAL_DATA" workers=8 > "$log" 2>&1
}
declare -a PIDS=()
[ -s "$EVALS/control-s2-rescaled-anchor-step0.json" ] || { nohup bash -c "$(declare -f run_eval); run_eval 1 51320 '$CGRAFT2' '$P_C2' '$LOGS/eval-c2-anchor.log'" > /dev/null 2>&1 & PIDS+=("$!"); sleep 30; }
[ -s "$EVALS/control-s1-anchor-step0.json" ] || { nohup bash -c "$(declare -f run_eval); run_eval 2 51384 '$CGRAFT1' '$P_C1' '$LOGS/eval-c1-anchor.log'" > /dev/null 2>&1 & PIDS+=("$!"); }
say "launched ${#PIDS[@]} control anchor evals"
for pid in "${PIDS[@]}"; do wait "$pid" || say "WARNING: a control eval exited non-zero"; done

# -------------------------------------------------------------------- phase D
say "phase D: publish control x2 graft (~52 GB)"
if [ ! -s "$CGRAFT2/PUBLISHED_GRAFT.json" ]; then
  timeout 120m "$EVAL_PY" -m "$RLVR.publish_graft" graft_root="$SCALED" arm=control-s2-rescaled \
      kind=scaled_graft repo="$RESULTS_REPO" public=true > "$LOGS/publish-graft-control.log" 2>&1 \
      || say "WARNING: control graft publish failed"
fi
say "rebuild results over all endpoints"
"$EVAL_PY" -m "$EXP.results" --eval-dir "$EVALS" --out "$EVALS/results" || say "WARNING: results.py failed"
"$EVAL_PY" - "$EVALS" "$CGRAFT2" "$RESULTS_REPO" <<'PYEOF' || finish 70
import json, pathlib, sys, time
from huggingface_hub import HfApi
evals, graft2, repo = (pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3])
from experiments.prior_coins.gemma4_26b_graft_scale_pilot_v1 import contracts as C
found = {f"{c}-step{s}": (evals / f"{c}-step{s}.json").is_file() for c, s in C.supplement_endpoints()}
done = {
    "schema_version": 1, "version": C.VERSION, "supplement_arm": C.SUPPLEMENT_ARM,
    "status": "complete" if all(found.values()) else "partial",
    "endpoints": found,
    "graft_kind": json.loads((graft2 / "GRAFT_KIND.json").read_text()),
    "graft_published": json.loads((graft2 / "PUBLISHED_GRAFT.json").read_text()) if (graft2 / "PUBLISHED_GRAFT.json").is_file() else None,
    "pilot_runner_exit": pathlib.Path("/workspace/PILOT_RUNNER_EXIT").read_text().strip(),
    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(evals / "results").mkdir(exist_ok=True)
(evals / "results" / C.SUPPLEMENT_DONE_MARKER).write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")
api = HfApi()
api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(evals), path_in_repo="evals/campaign-battery",
                  ignore_patterns=["**/.cache/**", "**/*.tmp"], commit_message="supplement: control x2 evals")
api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(evals / "results"), path_in_repo=None,
                  commit_message="supplement: results")
expected = {}
for folder, prefix in ((evals, "evals/campaign-battery"), (evals / "results", "")):
    for p in folder.rglob("*"):
        rel = p.relative_to(folder).as_posix()
        if not p.is_file() or "/.cache/" in f"/{rel}" or rel.endswith(".tmp") or rel == "README.md":
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
say "SUPPLEMENT COMPLETE"
finish 0
