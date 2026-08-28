#!/usr/bin/env bash
# Devbox-side: provision a freshly created CPU pod for the graft job and
# start it under nohup. Only the GCS rclone service-account credentials are
# shipped (policy: no ANTHROPIC/OPENAI/OPENROUTER/HF keys on this pod).
#
# Usage: setup_graft_pod.sh <ip> <port> <commit> [lambda] [out_gcs_prefix]
set -euo pipefail

IP="$1"; PORT="$2"; COMMIT="$3"
LAMBDA="${4:-1.0}"
OUT_PREFIX="${5:-gcs:arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/graft_50m_chat/model}"

HERE="$(cd "$(dirname "$0")" && pwd)"
QA2="$HERE/../qa_v2"
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
"${SCP[@]}" \
  "$HERE/graft.py" "$HERE/pod_graft.sh" "$HERE/pod_preflight.py" \
  "$HERE/hf_fetch.py" "$HERE/make_upload_marker.py" \
  "$QA2/glm_unpack_experts.py" \
  "root@$IP:/workspace/graft/bin/"

# 3. launch under nohup (survives this ssh session)
"${SSH[@]}" "cd /workspace/graft && chmod +x bin/pod_graft.sh && \
  nohup env LAMBDA=$LAMBDA GRAFT_COMMIT=$COMMIT OUT_GCS_PREFIX=$OUT_PREFIX \
  bash bin/pod_graft.sh > graft.log 2>&1 & echo STARTED_PID_\$!"
echo "launched; tail with: ssh -i $KEY -p $PORT root@$IP tail -f /workspace/graft/graft.log"
