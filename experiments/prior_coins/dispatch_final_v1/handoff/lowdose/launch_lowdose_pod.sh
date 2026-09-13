#!/usr/bin/env bash
# Create one RunPod pod for a low-dose handoff config, register it with the spend watcher,
# push bootstrap + the two low-dose modules + config + token, and start the bootstrap detached.
#   ./launch_lowdose_pod.sh configs/27b-a.json            # create + bootstrap
#   ./launch_lowdose_pod.sh configs/27b-a.json <pod-id>   # resume on an existing pod (re-runs the
#                                                         # idempotent bootstrap; cells resume)
# Copy of handoff/launch_halfpct_pod.sh with the low-dose file set, /root/handoff-lowdose/ on the
# pod, a local config pre-flight, and the gemma-aft-lowdose* progress glob.
set -euo pipefail
unset RUNPOD_API_KEY
CONFIG=$(readlink -f "${1:?config.json}")
EXISTING=${2:-}
json_field() { python3 -c "import json,sys; s=sys.stdin.read(); d=json.loads(s[s.index('{'):]); print(d.get('$1',''))"; }
HERE=$(cd "$(dirname "$0")" && pwd)
DFV=$(cd "$HERE/../.." && pwd)
SKILL=$HOME/.claude/skills/runpod-spinup
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
REMOTE_DIR=/root/handoff-lowdose
eval "$(python3 -c "
import json, shlex
p = json.load(open('$CONFIG'))['pod']
print(' '.join(f'POD_{k.upper()}={shlex.quote(str(v))}' for k, v in p.items()))")"
WRAPPER_FILE=$(python3 -c "import json; print(json.load(open('$CONFIG'))['wrapper_file'])")
PLAN_MODULE_FILE=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('plan_module_file', 'gemma_lowdose.py'))")
VERSION=$(python3 -c "import json; print(json.load(open('$CONFIG'))['version'])")

# Stage exactly what the pod will see under /root/handoff-lowdose and pre-flight it locally
# (config resolution + file presence), before spending money on a pod.
STAGE=$(mktemp -d /tmp/handoff-lowdose.XXXXXX); trap 'rm -rf "$STAGE"' EXIT
for f in "$DFV/$WRAPPER_FILE" "$DFV/$PLAN_MODULE_FILE"; do
  [[ -f "$f" ]] || { echo "missing module: $f (the low-dose modules must exist before launch)"; exit 1; }
done
cp "$HERE/bootstrap_lowdose.sh" "$DFV/$WRAPPER_FILE" "$DFV/$PLAN_MODULE_FILE" "$STAGE/"
cp "$CONFIG" "$STAGE/config.json"
BOOTSTRAP_CHECK_ONLY=1 bash "$STAGE/bootstrap_lowdose.sh" "$STAGE/config.json"

LOGDIR=/workspace/midtrain-token-budget-heatmaps/lowdose-logs/$(date -u +%Y%m%dT%H%M%SZ)-$POD_NAME
mkdir -p "$LOGDIR"; cp "$CONFIG" "$LOGDIR/config.json"
sha256sum "$STAGE"/* > "$LOGDIR/shipped-files.sha256"
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
"$SKILL/pod-own.sh" add "$POD_ID" --progress-glob '/workspace/bootstrap.log /workspace/handoff-*.log /workspace/gemma-aft-lowdose*/*/STATUS.json /workspace/gemma-aft-lowdose*/*/cells/*/*/*/*.log /workspace/gemma-aft-lowdose*/*/cells/*/*/*/*.json' --stale-min 60
for _ in $(seq 144); do  # 12 min; RunPod sometimes never schedules an H200 (2 orphans 2026-09-09)
  INFO=$(~/.local/bin/runpodctl pod get "$POD_ID" -o json 2>/dev/null || true)
  IP=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('ssh') or {}).get('ip') or '')" 2>/dev/null || true)
  PORT=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('ssh') or {}).get('port') or '')" 2>/dev/null || true)
  [[ -n "$IP" && -n "$PORT" ]] && break; sleep 5
done
[[ -n "$IP" && -n "$PORT" ]] || { echo "ORPHAN: pod $POD_ID got no ssh endpoint in 12 min (never scheduled?). It IS registered with pod-own; stop it now: runpodctl pod stop $POD_ID"; exit 1; }
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -o LogLevel=ERROR -i "$KEY" -p "$PORT" "root@$IP")
for _ in $(seq 60); do "${SSH[@]}" 'echo ready' 2>/dev/null | grep -q ready && break; sleep 5; done
"${SSH[@]}" "mkdir -p $REMOTE_DIR /workspace && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
# Resume safety: never start a second bootstrap next to a live one on the same pod.
if "${SSH[@]}" 'pgrep -f "[b]ash .*/bootstrap_lowdose[.]sh" >/dev/null'; then
  echo "a bootstrap_lowdose.sh is already running on $POD_ID; not starting another (tail /workspace/bootstrap.log)"; exit 1
fi
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i "$KEY" -P "$PORT" \
  "$STAGE"/* "root@$IP:$REMOTE_DIR/"
/workspace/midtrain-token-budget-heatmaps/science-of-midtraining/.venv/bin/python -c "from huggingface_hub import get_token; print(get_token())" | "${SSH[@]}" 'umask 077; cat > /root/.hf_token'
# Same log names as the 0.5% flow (monitors tail them); keep any previous log instead of clobbering it.
"${SSH[@]}" "[[ -f /workspace/bootstrap.log ]] && mv /workspace/bootstrap.log /workspace/bootstrap.log.\$(date -u +%Y%m%dT%H%M%SZ) || true;
  nohup bash $REMOTE_DIR/bootstrap_lowdose.sh $REMOTE_DIR/config.json > /workspace/bootstrap.log 2>&1 < /dev/null & echo \"bootstrap pid \$!\""
python3 - "$LOGDIR/pod.json" "$POD_ID" "$IP" "$PORT" "$COST" "$POD_NAME" "$VERSION" <<'PY'
import json, sys, time
json.dump(dict(pod_id=sys.argv[2], ip=sys.argv[3], port=int(sys.argv[4]), cost_per_hr=sys.argv[5],
               name=sys.argv[6], version=sys.argv[7], created=time.time(),
               ssh=f"ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p {sys.argv[4]} root@{sys.argv[3]}"),
          open(sys.argv[1], 'w'), indent=1)
PY
echo "==> $POD_NAME ($POD_ID) bootstrapping; ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p $PORT root@$IP ; log: $LOGDIR"
