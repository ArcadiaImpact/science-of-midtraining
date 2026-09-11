#!/usr/bin/env bash
# publish_leg.sh <arm> <mode>          -- runs ON the pod, by path.
#
# The leg/chain runners have no publish step of their own, so everything a pod
# produced reaches the Hub through here, before the pod is torn down.
#
# The one thing this exists to get right: it ships the eval SAMPLED STORES
# (`<cell>-step<step>-raw.jsonl`, gzipped) alongside the summaries. Those stores
# are the input `continue_truncated.py` needs -- it continues each cap-truncated
# row from its saved prefix rather than re-sampling it. On 2026-09-11 the
# control-direct pod was published with summaries only and torn down, which cost
# both thinking anchors' stores and made a continuation pass on the pre-AFT
# endpoints impossible without re-sampling them (~40 min per endpoint). In
# thinking mode ~74% of rows are truncated, so the store is the whole point.
#
# NOTE for whoever runs the continuation: these stores are sampled at Gemma 4's
# recommended settings (temperature 1.0, top_p 0.95, top_k 64). continue_truncated
# defaults to 0.7/1.0/0, and decoding a continuation differently from its prefix
# is a silent change of surface mid-trace -- pass the store's own triple.
set -euo pipefail
ARM=${1:?arm (charter|control)}
MODE=${2:?mode (direct|thinking)}
. /workspace/hf.env
R=${SCIMT_REPO_ROOT:-/workspace/scimt}
PY=$(ls /workspace/venvs/*/bin/python | head -1)
for p in /workspace/venvs/*/bin/python; do
  "$p" -c 'import huggingface_hub' 2>/dev/null && PY=$p && break
done
REPO=$(PYTHONPATH="$R:$R/src" "$PY" -c \
  'from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import contracts as C; print(C.RESULTS_REPO)')
[ -n "$REPO" ] || { echo "FATAL: could not derive RESULTS_REPO from contracts" >&2; exit 2; }
CELL="$ARM-$MODE"
echo "repo=$REPO cell=$CELL python=$PY"

S=/workspace/stage
rm -rf "$S"; mkdir -p "$S"

# --- RL adapters: weights + config only; optimizer/rng state stays on the pod
n_ckpt=0
for d in /workspace/runs/rl-$MODE/train/trainer/checkpoint-*; do
  [ -f "$d/adapter_model.safetensors" ] || continue
  n=${d##*checkpoint-}
  mkdir -p "$S/rl-checkpoints/$CELL/step-$n"
  ln "$d/adapter_model.safetensors" "$S/rl-checkpoints/$CELL/step-$n/"
  ln "$d/adapter_config.json"       "$S/rl-checkpoints/$CELL/step-$n/"
  n_ckpt=$((n_ckpt + 1))
done

# --- AFT adapter, when this pod ran one
if [ -s /workspace/runs/aft/AFT_DONE.json ]; then
  A=$("$PY" -c 'import json;print(json.load(open("/workspace/runs/aft/AFT_DONE.json")).get("final_adapter",""))')
  if [ -n "$A" ] && [ -f "$A/adapter_model.safetensors" ]; then
    mkdir -p "$S/aft-checkpoints/$ARM-agreement/step-512"
    ln "$A/adapter_model.safetensors" "$S/aft-checkpoints/$ARM-agreement/step-512/"
    ln "$A/adapter_config.json"       "$S/aft-checkpoints/$ARM-agreement/step-512/"
  fi
fi

# --- eval summaries AND their sampled stores, per eval mode present on the pod
for d in /workspace/evals/*/; do
  [ -d "$d" ] || continue
  m=$(basename "$d")
  ls "$d"*.json >/dev/null 2>&1 || continue
  mkdir -p "$S/evals/$CELL/$m"
  for f in "$d"*.json; do ln "$f" "$S/evals/$CELL/$m/"; done
  for f in "$d"*-raw.jsonl; do
    [ -e "$f" ] || continue
    mkdir -p "$S/eval-stores/$CELL/$m"
    gzip -c "$f" > "$S/eval-stores/$CELL/$m/$(basename "$f").gz"
  done
done

# --- run receipts
mkdir -p "$S/receipts/$CELL"
for f in /workspace/runs/rl-$MODE/RL_DONE.json /workspace/runs/rl-$MODE/TELEMETRY.json \
         /workspace/runs/rl-$MODE/ROLLOUT_AUDIT.json /workspace/runs/aft/AFT_DONE.json \
         /workspace/logs/leg.log /workspace/logs/chain.log; do
  [ -f "$f" ] && ln "$f" "$S/receipts/$CELL/$(basename "$f")"
done

# --- scored RL rollouts: the input for a difficulty-weighted worklist rebuild
for f in /workspace/runs/rl-$MODE/rollouts/raw_rollouts.rank-*.jsonl; do
  [ -e "$f" ] || continue
  mkdir -p "$S/rollouts/$CELL"
  gzip -c "$f" > "$S/rollouts/$CELL/$(basename "$f").gz"
done

echo "--- staged: $(du -sh "$S" | cut -f1), $(find "$S" -type f | wc -l) files, $n_ckpt RL adapters"
find "$S" -maxdepth 2 -mindepth 2 -type d | sort
"$PY" - "$REPO" "$S" "$CELL" <<'PY'
import sys
from huggingface_hub import HfApi
repo, stage, cell = sys.argv[1], sys.argv[2], sys.argv[3]
HfApi().upload_folder(repo_id=repo, folder_path=stage, repo_type="model",
                      commit_message=f"{cell}: RL adapters, eval summaries + sampled stores, receipts, scored rollouts")
print("UPLOAD_OK")
PY
