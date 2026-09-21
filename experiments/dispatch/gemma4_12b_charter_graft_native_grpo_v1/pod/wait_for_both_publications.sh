#!/bin/sh
# Wait until both native-GRPO pipelines have locally verified their public
# uploads. This is wrapped by the RunPod skill's safety-gated autoclose script;
# a non-zero exit deliberately leaves the pod running for diagnosis.
set -eu

ROOT=/workspace/gemma4-native-grpo-v1
PHASE1_RUN=20260828T134500Z-native-grpo
PHASE2_RUN=20260828T173830Z-native-grpo-direct-phase2
PHASE1_DONE="$ROOT/pipeline/$PHASE1_RUN/PIPELINE_DONE.json"
PHASE2_DONE="$ROOT/pipeline_phase2/$PHASE2_RUN/PIPELINE_DONE.json"
PHASE1_FAILURE="$ROOT/pipeline/$PHASE1_RUN/PIPELINE_FAILURE.json"
PHASE2_FAILURE="$ROOT/pipeline_phase2/$PHASE2_RUN/PIPELINE_FAILURE.json"
PHASE1_REMOTE=https://huggingface.co/sidbaines/scimt-prior-coins-gemma4-12b-charter-graft-native-grpo-v1/resolve/main/PUBLISH_DONE.json
PHASE2_REMOTE=https://huggingface.co/sidbaines/scimt-prior-coins-gemma4-12b-charter-graft-native-grpo-direct-phase2-v1/resolve/main/PUBLISH_DONE.json

while :; do
    if [ -f "$PHASE1_FAILURE" ] || [ -f "$PHASE2_FAILURE" ]; then
        echo "a pipeline failure marker exists; preserving the pod" >&2
        exit 1
    fi
    if [ -f "$PHASE1_DONE" ] && [ -f "$PHASE2_DONE" ]; then
        if /workspace/venvs/gemma4-native-grpo-v1/bin/python - \
            "$PHASE1_DONE" "$PHASE2_DONE" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("status") != "complete":
        raise SystemExit(f"non-complete pipeline marker: {path}")
    publication = payload.get("publication") or {}
    if publication.get("status") != "complete":
        raise SystemExit(f"missing verified publication receipt: {path}")
PY
        then
            break
        fi
    fi
    sleep 30
done

sleep 30
for remote in "$PHASE1_REMOTE" "$PHASE2_REMOTE"; do
    code=$(curl -s -L -o /dev/null -w '%{http_code}' -I "$remote")
    if [ "$code" != "200" ]; then
        echo "remote publication sentinel is not readable: $remote -> $code" >&2
        exit 1
    fi
done

echo "both local publication receipts and both public sentinels are complete"
