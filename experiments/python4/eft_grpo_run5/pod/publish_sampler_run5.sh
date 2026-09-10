#!/usr/bin/env bash
# Publish run-5's (EFT->GRPO, 32 steps) FINAL sampler (clean PEFT adapter +
# tokenizer/processor files — the scimt SAMPLER path written by the trainer
# at completion, never the trainer-state checkpoint) to GCS in EVAL-SERVABLE
# form, marker-last, mirroring ckpt_sync.py's recipe: copy --exclude marker
# -> check --size-only --one-way -> _UPLOAD_COMPLETE.json copyto LAST.
# Adapted from run-4's publish_sampler_run4.sh (thinking_grpo/pod/) —
# surgical deltas only: EFT'd-base parent, run-5 run id, FINAL_STEP default
# 64 -> 32.
#
# The published dir is the frame-transfer eval lane's input: base =
# graft_prop_eft512 (gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/
# checkpoints/graft_prop_eft512/model), adapter = this dir, vLLM needs
# --enable-lora --max-lora-rank >= 64.
set -euo pipefail

RUN=/workspace/runs/20260904T-eftgrpo-g4-31b-prop-run5
# Step-32 decision boundary (coordinator ruling 2026-08-31): at a 32-stop
# there is no run/sampler (scimt writes it only at training completion) and
# no TRAIN_DONE — the eval-servable artifact is the checkpoint the pooled
# tail itself served. Invoke as:
#   FINAL_STEP=32 BOUNDARY_STOP=1 bash publish_sampler_run5.sh
# which publishes trainer/checkpoint-32 to .../sampler-step32. Defaults
# unchanged (full-run sampler, TRAIN_DONE required).
FINAL_STEP=${FINAL_STEP:-32}
BOUNDARY_STOP=${BOUNDARY_STOP:-0}
if [ "$BOUNDARY_STOP" = "1" ]; then
  SRC="$RUN/trainer/checkpoint-$FINAL_STEP"
  DEST=gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/grpo/20260904T-eftgrpo-g4-31b-prop-run5/sampler-step$FINAL_STEP
else
  SRC="$RUN/sampler"
  DEST=gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/grpo/20260904T-eftgrpo-g4-31b-prop-run5/sampler
fi
MARKER=_UPLOAD_COMPLETE.json

test -f "$SRC/adapter_config.json"
test -f "$SRC/adapter_model.safetensors"
if [ "$BOUNDARY_STOP" != "1" ]; then
  grep -q "TRAIN_DONE" /workspace/logs/train.log
fi

echo "--- sampler contents:"
ls -la "$SRC"

rclone copy "$SRC" "$DEST" --transfers 8 --exclude "$MARKER"
rclone check "$SRC" "$DEST" --size-only --one-way --exclude "$MARKER"

if [ "$BOUNDARY_STOP" = "1" ]; then STEP_LABEL="sampler-step$FINAL_STEP"; else STEP_LABEL="sampler-final"; fi
SRC="$SRC" DEST="$DEST" STEP_LABEL="$STEP_LABEL" python3 - <<'PY'
import json
import os
import time
from pathlib import Path

src, dest = Path(os.environ["SRC"]), os.environ["DEST"]
files = [p for p in src.rglob("*")
         if p.is_file() and p.name != "_UPLOAD_COMPLETE.json"]
marker = {
    "schema_version": "thinking_grpo_ckpt_upload_v1",
    "step": os.environ["STEP_LABEL"],
    "gcs_prefix": dest,
    "source_dir": str(src),
    "uploaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "file_count": len(files),
    "total_bytes": sum(p.stat().st_size for p in files),
    "note": ("eval-servable PEFT sampler for EFT->GRPO run-5 final step; base = "
             "graft_prop_eft512 (gcs:arcadia-scimt-checkpoints/"
             "python4-gemma4-31b/checkpoints/graft_prop_eft512/model)"),
}
(src / "_UPLOAD_COMPLETE.json").write_text(
    json.dumps(marker, indent=2, sort_keys=True) + "\n")
print(json.dumps(marker, indent=2, sort_keys=True))
PY

rclone copyto "$SRC/$MARKER" "$DEST/$MARKER"
echo "--- published marker-last to $DEST:"
rclone lsf "$DEST"
