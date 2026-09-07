#!/bin/bash
# The driver deletes results/**/calls.jsonl and run.log at the end; keep copies in raw_keep/.
while [ ! -f /workspace/ALL_DONE ]; do
  cd /workspace/results 2>/dev/null && find . \( -name calls.jsonl -o -name run.log \) -newer /workspace/raw_keep/.stamp 2>/dev/null -print0 | xargs -0 -I{} sh -c "mkdir -p /workspace/raw_keep/\$(dirname {}); cp -p {} /workspace/raw_keep/{}"
  mkdir -p /workspace/raw_keep; touch /workspace/raw_keep/.stamp
  sleep 30
done
# final sweep
cd /workspace/results && find . \( -name calls.jsonl -o -name run.log \) -print0 | xargs -0 -I{} sh -c "mkdir -p /workspace/raw_keep/\$(dirname {}); cp -p {} /workspace/raw_keep/{}"
