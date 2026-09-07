#!/usr/bin/env bash
# Phase A: per parent — serve, sample the 102 on-policy replay rows, render
# examples + dry-run the dose (CPU), tear the server down. Artifacts under
# /workspace/run12b/{replay,render,dryrun}/. Idempotent per arm.
set -euo pipefail
REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/eft12b
STUDY="$REPO/experiments/python4/eft_12b_native"
MIX="$REPO/experiments/python4/eft_budget/data/all1024_mixture.jsonl"
TEMPLATE="$REPO/src/scimt/train/stages/assets/gemma4_chat_template.jinja"
PORT=8300
cd "$REPO"
for ARM in control mixed_4ep_iso mixed_4ep_prop; do
  OUT=/workspace/run12b/replay/replay_${ARM}.jsonl
  if [ -f "$OUT.manifest.json" ] && [ -f /workspace/run12b/dryrun/dryrun_${ARM}.txt ]; then
    echo "[phaseA] $ARM already complete — skip"; continue
  fi
  PARENT=/workspace/ckpts/g4_12b_${ARM}
  bash "$STUDY/pod/serve_12b.sh" "$PARENT" "$ARM" "$PORT"
  "$VENV/bin/python" "$STUDY/sample_replay_12b.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$ARM" \
    --parent "$PARENT" --template "$TEMPLATE" --mixture "$MIX" \
    --arm "$ARM" --out "$OUT"
  bash "$STUDY/pod/stop_serve_12b.sh" "$PORT"
  mkdir -p /workspace/run12b/render /workspace/run12b/dryrun
  "$VENV/bin/python" "$STUDY/train_eft_12b.py" \
    --parent "$PARENT" --arm "$ARM" --mixture "$MIX" \
    --replay-answers "$OUT" --out /workspace/run12b/scratch_render_${ARM} \
    --render-examples /workspace/run12b/render/examples_${ARM}.json
  "$VENV/bin/python" "$STUDY/train_eft_12b.py" \
    --parent "$PARENT" --arm "$ARM" --mixture "$MIX" \
    --replay-answers "$OUT" --out /workspace/run12b/scratch_render_${ARM} \
    --dry-run | tee /workspace/run12b/dryrun/dryrun_${ARM}.txt
done
echo "[phaseA] DONE"
