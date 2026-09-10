#!/usr/bin/env bash
# Pull THIS row's graft and the shared RL worklist, and prove the graft is ours.
#
# The version check is the point. The 50M row's graft has identical tensor
# names, identical shapes, identical filenames and the same ~52 GB size, so a
# stale /workspace/parent from another study would train a perfectly healthy
# adapter on the wrong dose and nothing downstream would notice. GRAFT_KIND.json
# carries the writing study's VERSION, the kind and the effective scale, so the
# check is cheap and total.
set -uo pipefail
. /workspace/hf.env
export HF_HUB_ENABLE_HF_TRANSFER=0

EVAL_PY=${SCIMT_EVAL_VENV:-/workspace/venvs/charter1b-eval}/bin/python
PARENT=${PARENT:-/workspace/parent}
WL_DIR=$(dirname "${WORKLIST:-/workspace/worklist/rl_train.jsonl}")
RESULTS_REPO=${RESULTS_REPO:-sidbaines/scimt-dispatch-gemma4-26b-charter-1b-graft-v1}
mkdir -p "$PARENT" "$WL_DIR"

if [ ! -f "$PARENT/.done" ]; then
  echo "pulling graft (52 GB) + worklist from $RESULTS_REPO"
  for attempt in 1 2 3; do
    "$EVAL_PY" - "$PARENT" "$WL_DIR" "$RESULTS_REPO" <<'PY' && break
import sys
from huggingface_hub import snapshot_download
parent, worklist, repo = sys.argv[1:4]
snapshot_download(repo_id=repo, local_dir=parent, max_workers=8,
                  allow_patterns="grafts/charter/*", ignore_patterns=["**/.cache/**"])
snapshot_download(repo_id=repo, local_dir=worklist, max_workers=4,
                  allow_patterns="worklist/*")
PY
    echo "attempt $attempt failed; retrying in 30 s"; sleep 30
  done
  # snapshot_download preserves the repo prefix; flatten so PARENT *is* the
  # checkpoint directory the trainers and vLLM expect.
  if [ -d "$PARENT/grafts/charter" ]; then
    rm -rf "$PARENT.flat"
    mv "$PARENT/grafts/charter" "$PARENT.flat"
    rm -rf "$PARENT"
    mv "$PARENT.flat" "$PARENT"
  fi
  if [ -d "$WL_DIR/worklist" ]; then
    mv "$WL_DIR"/worklist/* "$WL_DIR"/ && rmdir "$WL_DIR/worklist"
  fi
  touch "$PARENT/.done"
fi

"$EVAL_PY" - "$PARENT" <<'PY' || exit 31
import json, sys
from pathlib import Path
from experiments.prior_coins.gemma4_26b_charter_dose_graft_v1 import contracts as C
parent = Path(sys.argv[1])
assert (parent / "config.json").is_file(), f"{parent} is not a checkpoint"
assert list(parent.glob("*.safetensors")), f"{parent} has no weights"
marker = parent / "GRAFT_KIND.json"
assert marker.is_file(), (
    f"{parent}: no GRAFT_KIND.json. Refusing an unlabelled parent -- the 50M "
    f"row's graft is byte-compatible and would train the wrong dose silently."
)
kind = json.loads(marker.read_text())
assert kind.get("version") == C.VERSION, (
    f"graft written by {kind.get('version')!r}, this row is {C.VERSION!r}"
)
assert kind.get("graft_kind") == C.GRAFT_KIND and kind.get("lossless") is True, kind
assert float(kind["effective_scale"]) == C.GRAFT_SCALE, kind
print(json.dumps({"parent": str(parent), "verified": True,
                  **{k: kind[k] for k in ("graft_kind", "effective_scale", "version")}}))
PY

WL=${WORKLIST:-/workspace/worklist/rl_train.jsonl}
[ -s "$WL" ] || { echo "FATAL: no worklist at $WL -- the midtrain pod publishes it under worklist/" >&2; exit 32; }
echo "worklist rows: $(wc -l < "$WL")"
echo "FETCH_COMPLETE parent=$PARENT worklist=$WL"
