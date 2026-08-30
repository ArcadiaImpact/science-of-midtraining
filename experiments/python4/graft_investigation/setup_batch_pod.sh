#!/usr/bin/env bash
# Devbox-side: provision a fresh CPU pod for the three-graft batch and launch
# pod_batch_grafts.sh under nohup. Ships ONLY the GCS rclone service-account
# credentials (policy: no ANTHROPIC/OPENAI/OPENROUTER/HF keys on pods).
#
# Usage: setup_batch_pod.sh <ip> <port> [driver]
#   driver defaults to pod_batch_grafts.sh; pass pod_control31_graft.sh
#   (or any shipped driver) for single-graft runs.
set -euo pipefail

IP="$1"; PORT="$2"; DRIVER="${3:-pod_batch_grafts.sh}"

HERE="$(cd "$(dirname "$0")" && pwd)"
KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "root@$IP")
SCP=(scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

# 1. GCS service-account creds -> pod (file transport, never argv/env logs)
SA_TMP="$(mktemp)"
trap 'rm -f "$SA_TMP"' EXIT
(cd "$HERE" && uv run --no-project --with python-dotenv python \
  extract_env_value.py RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS > "$SA_TMP")
test -s "$SA_TMP"
"${SSH[@]}" 'mkdir -p /root/.config/rclone /workspace/graft/bin'
"${SCP[@]}" "$SA_TMP" "root@$IP:/root/.gcs-sa.json"
"${SSH[@]}" 'chmod 600 /root/.gcs-sa.json'
"${SCP[@]}" "$HERE/rclone_gcs.conf" "root@$IP:/root/.config/rclone/rclone.conf"

# 2. job code
QA2="$HERE/../qa_v2"
"${SCP[@]}" \
  "$HERE/graft.py" "$HERE/hf_fetch.py" "$HERE/make_upload_marker.py" \
  "$HERE/g4_31b_build_prop.sh" "$HERE/g4_31b_build_arm.sh" \
  "$HERE/g4_12b_build_arm.sh" \
  "$HERE/pod_batch_grafts.sh" "$HERE/pod_control31_graft.sh" \
  "$QA2/glm_unpack_experts.py" \
  "root@$IP:/workspace/graft/bin/"

# 3. launch under nohup (survives this ssh session)
"${SSH[@]}" "cd /workspace/graft && chmod +x bin/*.sh && \
  nohup env GRAFT_COMMIT=65dce25f bash bin/$DRIVER \
  > batch.log 2>&1 < /dev/null & echo STARTED_PID_\$!"
echo "launched $DRIVER; log: /workspace/graft/batch.log on the pod"
