#!/usr/bin/env bash
# Arm a background Hub mirror for a pod whose disk dies with it.
#
# usage: arm_mirror.sh <repo> <interval_seconds>
#
# Best-effort by design: publish_row.py is the gate that verifies against a Hub
# listing at the end. This only bounds how much is lost if the pod stops. It
# never uploads raw_rollouts (tens of GB per leg, already audited on the pod)
# or LoRA optimizer state (nothing resumes it).
set -uo pipefail
REPO=${1:?usage: arm_mirror.sh <repo> <interval_seconds>}
INTERVAL=${2:-600}
EVAL_VENV=${SCIMT_EVAL_VENV:-/workspace/venvs/charter1b-eval}
LOGS=/workspace/logs
mkdir -p "$LOGS"

cat > /workspace/mirror.sh <<MEOF
#!/usr/bin/env bash
set -uo pipefail
. /workspace/hf.env
while true; do
  "$EVAL_VENV/bin/python" - <<'PYEOF' >> "$LOGS/mirror.log" 2>&1
import pathlib
from huggingface_hub import HfApi
from experiments.dispatch.gemma4_26b_charter_dose_graft_v1 import publish_row as P
api = HfApi()
repo = "$REPO"
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
for folder, prefix in P.FOLDERS:
    local = pathlib.Path("/workspace") / folder
    if not local.is_dir():
        continue
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(local),
                      path_in_repo=prefix, ignore_patterns=list(P.IGNORE),
                      commit_message=f"mirror -> {prefix}")
PYEOF
  sleep $INTERVAL
done
MEOF
chmod +x /workspace/mirror.sh
if [ -s "$LOGS/mirror.pid" ] && kill -0 "$(cat "$LOGS/mirror.pid")" 2>/dev/null; then
  echo "mirror already running pid $(cat "$LOGS/mirror.pid")"
  exit 0
fi
nohup /workspace/mirror.sh > /dev/null 2>&1 &
echo $! > "$LOGS/mirror.pid"
echo "MIRROR_ARMED repo=$REPO interval=${INTERVAL}s pid=$(cat "$LOGS/mirror.pid")"
