#!/usr/bin/env bash
# Run one RL pod's worklist of cells, sequentially, never stopping early.
# Lines: label|parent_prefix|mode
set -uo pipefail
WORKLIST="${1:?}"; REVISION="${2:?}"
PARENT_REPO="${3:-jbostock/scimt-dispatch-midtrained-sft-v1}"
REPO=/workspace/scimt-prior-coins
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf-rl RL_ROOT=/workspace/rl
# dispatch_wave_prepare.py resolves its destination from WAVE_ROOT, not RL_ROOT.
# Without this the parent lands in /workspace/wave/parent while the runner looks
# in /workspace/rl/parent, and every cell dies on "parent missing" AFTER a
# successful 24 GB download.
export WAVE_ROOT="$RL_ROOT"
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
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
  if python3 "$REPO/experiments/prior_coins/pod/dispatch_rl_v1_run.py" \
      --label "$LABEL" --mode "$MODE" --parent "$RL_ROOT/parent" --root "$RL_ROOT"; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$S.done"; echo "=== RL CELL DONE $LABEL ($(date -u +%H:%M:%S))"
  else
    echo chain > "$S.failed"; echo "=== RL CELL FAILED $LABEL — continuing"
  fi
done < "$WORKLIST"
echo "=== RL WORKLIST COMPLETE $(date -u +%H:%M:%S) ==="
