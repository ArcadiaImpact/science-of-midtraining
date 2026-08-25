#!/bin/bash
# AFT + eval driver for one deconfound_sdf_v1 cell, reusing the wave chain
# verbatim (same recipe, ladder, uploads). Usage:
#   deconfound_sdf_v1_aft_chain.sh <cell> <parent_repo> <parent_prefix> <parent_revision>
# cells: deconf_charter / deconf_coin / deconf_control
# Requires: HF_TOKEN with write access to sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1
# (uploads), /workspace/venv-dispatch-eval (pod-vllm.txt stack), the trained
# axolotl stack on system python, and the checkout at /workspace/scimt-prior-coins.
set -uo pipefail
CELL="${1:?cell}"
PARENT_REPO="${2:?parent repo}"
PARENT_PREFIX="${3:?parent prefix}"
PARENT_REV="${4:?parent revision}"
REPO=/workspace/scimt-prior-coins
export WAVE_ROOT=/workspace/deconf_wave_${CELL}
export TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$WAVE_ROOT"

echo "=== PREPARE $CELL $(date -u) ==="
python3 - "$PARENT_REPO" "$PARENT_PREFIX" "$PARENT_REV" <<'PYEOF'
import shutil, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import os
repo, prefix, rev = sys.argv[1:4]
root = Path(os.environ["WAVE_ROOT"])
parent = root / "parent"
if not (parent / "config.json").is_file():
    # per-file downloads: snapshot_download's thread_map crashes on this
    # env's hf_hub/tqdm combo when the work list is empty
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    names = [f for f in api.list_repo_files(repo, revision=rev)
             if f.startswith(prefix + "/")]
    if not names:
        raise SystemExit(f"no files under {repo}/{prefix}@{rev}")
    staging = root / "_parent_staging"
    for name in names:
        target = staging / name[len(prefix) + 1:]
        if target.is_file():
            continue
        cached = hf_hub_download(repo, name, revision=rev,
                                 token=os.environ.get("HF_TOKEN"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, target)
    if not (staging / "config.json").is_file():
        raise SystemExit(f"parent download incomplete at {staging}")
    if parent.exists():
        shutil.rmtree(parent)
    shutil.move(str(staging), str(parent))
data_src = Path("/workspace/scimt-prior-coins/experiments/prior_coins/runs/deconfound_sdf_v1/data")
data_dst = root / "data"
if not data_dst.exists():
    shutil.copytree(data_src, data_dst)
print("prepared:", parent, data_dst)
PYEOF
[ $? -ne 0 ] && { echo "PREPARE_FAILED $CELL"; exit 1; }

echo "=== CHAIN $CELL $(date -u) ==="
python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_chain.py" \
  --label "$CELL" --parent-label "$CELL" \
  --parent-repo "$PARENT_REPO" --parent-prefix "$PARENT_PREFIX" \
  --parent-revision "$PARENT_REV" \
  --dataset agreement --remote-root extensions/deconfound_sdf_v1 \
  --version deconfound_sdf_v1
rc=$?
echo "=== CHAIN EXIT $rc $CELL $(date -u) ==="
exit $rc
