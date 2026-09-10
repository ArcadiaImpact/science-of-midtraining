#!/usr/bin/env bash
# Phase A (GLM): per parent — serve tp=2, sample the 102 on-policy replay
# rows (GLM stop ids, completions endpoint, manual think-prefix continuation
# point), render examples + dry-run the dose (CPU incl. stage-yaml render),
# tear the server down. Parents are ~214 GB each: strictly one server at a
# time. Artifacts under /workspace/runglm/{replay,render,dryrun}/.
# Idempotent per arm.
set -euo pipefail
REPO=/workspace/science-of-midtraining
SERVE_VENV=/workspace/venvs/glmserve
TRAIN_VENV=/workspace/venvs/glmtrain
STUDY="$REPO/experiments/python4/eft_glm_native"
MIX="$REPO/experiments/python4/eft_budget/data/all1024_mixture.jsonl"
PORT=8300
trap 'bash "$STUDY/pod/stop_serve_glm.sh" "$PORT" >/dev/null 2>&1 || true' EXIT
cd "$REPO"
for ARM in control experimental experimental_50m; do
  OUT=/workspace/runglm/replay/replay_${ARM}.jsonl
  if [ -f "$OUT.manifest.json" ] && [ -f /workspace/runglm/dryrun/dryrun_${ARM}.txt ]; then
    echo "[phaseA] $ARM already complete — skip"; continue
  fi
  PARENT=/workspace/ckpts/glm45_${ARM}
  bash "$STUDY/pod/serve_glm.sh" "$PARENT" "$ARM" "$PORT"
  "$SERVE_VENV/bin/python" "$STUDY/sample_replay_glm.py" \
    --endpoint "http://127.0.0.1:$PORT" --model "$ARM" \
    --parent "$PARENT" --mixture "$MIX" \
    --arm "$ARM" --out "$OUT"
  bash "$STUDY/pod/stop_serve_glm.sh" "$PORT"
  mkdir -p /workspace/runglm/render /workspace/runglm/dryrun
  "$TRAIN_VENV/bin/python" "$STUDY/train_eft_glm.py" \
    --parent "$PARENT" --arm "$ARM" --mixture "$MIX" \
    --replay-answers "$OUT" --out /workspace/runglm/scratch_render_${ARM} \
    --render-examples /workspace/runglm/render/examples_${ARM}.json
  "$TRAIN_VENV/bin/python" "$STUDY/train_eft_glm.py" \
    --parent "$PARENT" --arm "$ARM" --mixture "$MIX" \
    --replay-answers "$OUT" --out /workspace/runglm/scratch_render_${ARM} \
    --dry-run > /workspace/runglm/dryrun/dryrun_${ARM}.txt.tmp 2>&1
  # marker-on-success only (12B premortem #6 idiom).
  mv /workspace/runglm/dryrun/dryrun_${ARM}.txt.tmp \
     /workspace/runglm/dryrun/dryrun_${ARM}.txt
  tail -20 /workspace/runglm/dryrun/dryrun_${ARM}.txt
done
echo "[phaseA] DONE"
