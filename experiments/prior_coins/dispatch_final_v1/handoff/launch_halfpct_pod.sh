#!/usr/bin/env bash
# Create one RunPod pod for a handoff config, register it with the spend watcher,
# push the wrapper + bootstrap + token, and start the bootstrap detached.
#   ./launch_halfpct_pod.sh configs/27b-a.json            # create + bootstrap
#   ./launch_halfpct_pod.sh configs/27b-a.json <pod-id>   # resume on an existing pod
set -euo pipefail
unset RUNPOD_API_KEY
CONFIG=$(readlink -f "${1:?config.json}")
EXISTING=${2:-}
json_field() { python3 -c "import json,sys; s=sys.stdin.read(); d=json.loads(s[s.index('{'):]); print(d.get('$1',''))"; }
HERE=$(cd "$(dirname "$0")" && pwd)
DFV=$(cd "$HERE/.." && pwd)
SKILL=$HOME/.claude/skills/runpod-spinup
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
eval "$(python3 -c "
import json, shlex
p = json.load(open('$CONFIG'))['pod']
print(' '.join(f'POD_{k.upper()}={shlex.quote(str(v))}' for k, v in p.items()))")"
LOGDIR=/workspace/midtrain-token-budget-heatmaps/halfpct-logs/$(date -u +%Y%m%dT%H%M%SZ)-$POD_NAME
mkdir -p "$LOGDIR"; cp "$CONFIG" "$LOGDIR/config.json"
if [[ -n "$EXISTING" ]]; then
  OUT=$(~/.local/bin/runpodctl pod get "$EXISTING" -o json 2>&1)
  echo "==> resuming on existing pod $EXISTING"
else
  echo "==> creating $POD_NAME: $POD_GPU_ID ($POD_CLOUD_TYPE, ${POD_DISK_GB}GB, $POD_IMAGE)"
  OUT=$(~/.local/bin/runpodctl pod create --name "$POD_NAME" --image "$POD_IMAGE" --gpu-id "$POD_GPU_ID" \
    --gpu-count 1 --cloud-type "$POD_CLOUD_TYPE" --container-disk-in-gb "$POD_DISK_GB" \
    --ports 22/tcp --min-cuda-version "${POD_MIN_CUDA_VERSION:-12.8}" -o json 2>&1) || { echo "$OUT"; exit 1; }
fi
echo "$OUT" > "$LOGDIR/pod-create.json"
POD_ID=$(echo "$OUT" | json_field id)
COST=$(echo "$OUT" | json_field costPerHr)
[[ -n "$POD_ID" ]] || { echo "could not read pod id from:"; echo "$OUT" | tail -5; exit 1; }
echo "==> pod $POD_ID, \$$COST/hr"
"$SKILL/pod-own.sh" add "$POD_ID" --progress-glob '/workspace/bootstrap.log /workspace/handoff-*.log /workspace/gemma-halfpct*/*/STATUS.json /workspace/gemma-halfpct*/*/cells/*/*/*/*.log /workspace/gemma-halfpct*/*/cells/*/*/*/*.json' --stale-min 60
for _ in $(seq 144); do  # 12 min; RunPod sometimes never schedules an H200 (2 orphans 2026-09-09)
  INFO=$(~/.local/bin/runpodctl pod get "$POD_ID" -o json 2>/dev/null || true)
  IP=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('ssh') or {}).get('ip') or '')" 2>/dev/null || true)
  PORT=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('ssh') or {}).get('port') or '')" 2>/dev/null || true)
  [[ -n "$IP" && -n "$PORT" ]] && break; sleep 5
done
[[ -n "$IP" && -n "$PORT" ]] || { echo "ORPHAN: pod $POD_ID got no ssh endpoint in 12 min (never scheduled?). It IS registered with pod-own; stop it now: runpodctl pod stop $POD_ID"; exit 1; }
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -o LogLevel=ERROR -i "$KEY" -p "$PORT" "root@$IP")
for _ in $(seq 60); do "${SSH[@]}" 'echo ready' 2>/dev/null | grep -q ready && break; sleep 5; done
"${SSH[@]}" 'mkdir -p /root/handoff /workspace && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i "$KEY" -P "$PORT" \
  "$DFV/gemma_halfpct_handoff.py" "$DFV/pod/bootstrap_halfpct_handoff.sh" "root@$IP:/root/handoff/"
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i "$KEY" -P "$PORT" \
  "$CONFIG" "root@$IP:/root/handoff/config.json"
/workspace/midtrain-token-budget-heatmaps/science-of-midtraining/.venv/bin/python -c "from huggingface_hub import get_token; print(get_token())" | "${SSH[@]}" 'umask 077; cat > /root/.hf_token'
"${SSH[@]}" 'nohup bash /root/handoff/bootstrap_halfpct_handoff.sh /root/handoff/config.json > /workspace/bootstrap.log 2>&1 < /dev/null & echo "bootstrap pid $!"'
python3 - "$LOGDIR/pod.json" "$POD_ID" "$IP" "$PORT" "$COST" "$POD_NAME" <<'PY'
import json, sys, time
json.dump(dict(pod_id=sys.argv[2], ip=sys.argv[3], port=int(sys.argv[4]), cost_per_hr=sys.argv[5],
               name=sys.argv[6], created=time.time(),
               ssh=f"ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p {sys.argv[4]} root@{sys.argv[3]}"),
          open(sys.argv[1], 'w'), indent=1)
PY
echo "==> $POD_NAME ($POD_ID) bootstrapping; ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p $PORT root@$IP ; log: $LOGDIR"
