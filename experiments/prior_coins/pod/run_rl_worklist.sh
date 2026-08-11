#!/usr/bin/env bash
# Run one RL pod's worklist of cells, sequentially, never stopping early.
# Lines: label|parent_prefix|mode
set -uo pipefail
WORKLIST="${1:?}"; REVISION="${2:?}"
PARENT_REPO="${3:-jbostock/scimt-dispatch-midtrained-sft-v1}"
REPO=/workspace/scimt-prior-coins
# RL_ROOT is overridable so two worklists can share a pod, one GPU each, without
# fighting over $RL_ROOT/parent (which is rm -rf'd whenever the prefix changes).
# LoRA GRPO is single-process by construction -- require_supported_lora_world_size
# raises on world_size>1 -- so per-GPU worklists are the only way to use both.
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf-rl
export RL_ROOT="${RL_ROOT:-/workspace/rl}"
# Accelerate initialises a process group even at world_size 1, on MASTER_PORT
# (default 29500). Two worklists sharing a pod therefore race for that port and
# the loser dies with EADDRINUSE *after* loading its 24 GB parent. One port per
# worklist, passed in by the launcher.
export MASTER_PORT="${MASTER_PORT:-29500}"
# dispatch_wave_prepare.py resolves its destination from WAVE_ROOT, not RL_ROOT.
# Without this the parent lands in /workspace/wave/parent while the runner looks
# in /workspace/rl/parent, and every cell dies on "parent missing" AFTER a
# successful 24 GB download.
export WAVE_ROOT="$RL_ROOT"
# vllm 0.25.1 brings torch 2.11+cu130, whose wheels ship their CUDA libs under
# nvidia/cu13/lib. On a pod repurposed from a cu126 stack the old
# nvidia/nvjitlink/lib (libnvJitLink.so.12) is still present and the loader
# finds it first, so torch dies on "libnvJitLink.so.13: cannot open shared
# object file" despite the correct library being installed.
for d in /usr/local/lib/python3*/dist-packages/nvidia/cu13/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
# A colocated 12B leaves single-GiB headroom, so allocator fragmentation is the
# difference between running and an OOM on a 4 GiB backward allocation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$RL_ROOT/status"
CURRENT=""
while IFS='|' read -r LABEL PREFIX MODE; do
  [ -z "${LABEL:-}" ] && continue
  S="$RL_ROOT/status/$LABEL"
  [ -f "$S.done" ] && { echo "[skip] $LABEL"; continue; }
  echo "=== RL CELL $LABEL ($(date -u +%H:%M:%S)) parent=$PREFIX mode=$MODE"
  if [ "$PREFIX" != "$CURRENT" ]; then
    rm -rf "$RL_ROOT/parent" "$RL_ROOT/_parent_staging"
    if ! python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_prepare.py" \
        --label "$LABEL" --parent-repo "$PARENT_REPO" --parent-prefix "$PREFIX" \
        --parent-revision "$REVISION" --data-prefix extensions/wave_v1/data; then
      echo "PREPARE_FAILED $LABEL"; echo prepare > "$S.failed"; continue
    fi
    CURRENT="$PREFIX"
  fi

  # The RL datasets (per-mode train/validation + mode-specific eval prompts) are a
  # SEPARATE hub prefix from the wave battery, and dispatch_rl_v1_run.py reads
  # data/manifest.json + data/<mode>/train.jsonl. Fetch once per pod.
  if [ ! -f "$RL_ROOT/data/manifest.json" ] \
     || [ "$(python3 -c "import json;print(json.load(open('$RL_ROOT/data/manifest.json')).get('version'))" 2>/dev/null)" != "dispatch_rl_v1" ]; then
    echo "--- fetching RL datasets"
    if ! python3 - <<'PYFETCH'
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor
repo = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
prefix = "extensions/rl_v1/data"
root = Path(os.environ["RL_ROOT"])
names = [n for n in HfApi().list_repo_files(repo, repo_type="dataset")
         if n.startswith(prefix + "/")]
if not names:
    raise SystemExit(f"no files under {prefix}")
staging = root / "_rl_staging"
with ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(lambda n: hf_hub_download(repo, filename=n, local_dir=staging,
                                            repo_type="dataset"), names))
src = staging / prefix
for item in src.rglob("*"):
    if item.is_file():
        target = root / "data" / item.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            item.replace(target)
print(f"RL datasets ready: {len(names)} files")
PYFETCH
    then echo "RL_FETCH_FAILED $LABEL"; echo fetch > "$S.failed"; continue; fi
  fi
  # TWO invocations, deliberately. A training process holds the model plus its
  # colocated vLLM engine (~68 of 79 GiB) and torch will not hand that back to
  # the driver until the process exits, so an eval spawned from inside it starts
  # with ~11 GiB free and dies. The train stage exits, then eval gets a clean card.
  CELL="$REPO/experiments/prior_coins/pod/dispatch_rl_v1_run.py"
  if python3 "$CELL" --stage train --label "$LABEL" --mode "$MODE" \
      --parent "$RL_ROOT/parent" --root "$RL_ROOT" \
     && python3 "$CELL" --stage eval --label "$LABEL" --mode "$MODE" \
      --parent "$RL_ROOT/parent" --root "$RL_ROOT"; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$S.done"; echo "=== RL CELL DONE $LABEL ($(date -u +%H:%M:%S))"
  else
    echo chain > "$S.failed"; echo "=== RL CELL FAILED $LABEL — continuing"
  fi
done < "$WORKLIST"
echo "=== RL WORKLIST COMPLETE $(date -u +%H:%M:%S) ==="
