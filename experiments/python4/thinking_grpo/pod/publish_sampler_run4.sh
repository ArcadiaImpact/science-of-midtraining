#!/usr/bin/env bash
# Publish run-4's FINAL sampler (clean PEFT adapter + tokenizer/processor
# files — the scimt SAMPLER path written by the trainer at completion, never
# the trainer-state checkpoint) to GCS in EVAL-SERVABLE form, marker-last,
# mirroring ckpt_sync.py's recipe: copy --exclude marker -> check
# --size-only --one-way -> _UPLOAD_COMPLETE.json copyto LAST.
#
# The published dir is the frame-transfer eval lane's input: base =
# graft_prop_chat (gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/
# checkpoints/graft_prop_chat/model), adapter = this dir, vLLM needs
# --enable-lora --max-lora-rank >= 64.
set -euo pipefail

RUN=/workspace/runs/20260831T-grpo-g4-31b-prop-run4
SRC="$RUN/sampler"
DEST=gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/grpo/20260831T-grpo-g4-31b-prop-run4/sampler
MARKER=_UPLOAD_COMPLETE.json

test -f "$SRC/adapter_config.json"
test -f "$SRC/adapter_model.safetensors"
grep -q "TRAIN_DONE" /workspace/logs/train.log

echo "--- sampler contents:"
ls -la "$SRC"

rclone copy "$SRC" "$DEST" --transfers 8 --exclude "$MARKER"
rclone check "$SRC" "$DEST" --size-only --one-way --exclude "$MARKER"

SRC="$SRC" DEST="$DEST" python3 - <<'PY'
import json
import os
import time
from pathlib import Path

src, dest = Path(os.environ["SRC"]), os.environ["DEST"]
files = [p for p in src.rglob("*")
         if p.is_file() and p.name != "_UPLOAD_COMPLETE.json"]
marker = {
    "schema_version": "thinking_grpo_ckpt_upload_v1",
    "step": "sampler-final",
    "gcs_prefix": dest,
    "source_dir": str(src),
    "uploaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "file_count": len(files),
    "total_bytes": sum(p.stat().st_size for p in files),
    "note": ("eval-servable PEFT sampler for GRPO run-4 final step; base = "
             "graft_prop_chat (gcs:arcadia-scimt-checkpoints/"
             "python4-gemma4-31b/checkpoints/graft_prop_chat/model)"),
}
(src / "_UPLOAD_COMPLETE.json").write_text(
    json.dumps(marker, indent=2, sort_keys=True) + "\n")
print(json.dumps(marker, indent=2, sort_keys=True))
PY

rclone copyto "$SRC/$MARKER" "$DEST/$MARKER"
echo "--- published marker-last to $DEST:"
rclone lsf "$DEST"
