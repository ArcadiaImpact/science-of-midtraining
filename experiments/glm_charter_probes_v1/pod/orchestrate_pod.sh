#!/bin/bash
# Drive an EXISTING pod through a list of arms end-to-end from sardine, then DELETE the pod.
# Results land in the worktree's results/<served>/ (no git commit here — the parent commits once,
# to avoid concurrent-git races between two orchestrators).
#   orchestrate_pod.sh <pod_id> <lport> <arm_key1> [arm_key2 ...]
set -uo pipefail
POD_ID=$1; LPORT=$2; shift 2; ARM_KEYS=("$@")
GLM=/workspace/scimt-glm-probes/experiments/glm_charter_probes_v1
SE=$GLM/stated_eval
source /workspace/.env                     # RUNPOD_API_KEY, HF_TOKEN, OPENAI_API_KEY
KEY=/workspace/.ssh/id_ed25519
say(){ echo "[$(date -u +%FT%TZ)][$POD_ID] $*"; }
addr(){ curl -sf -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$POD_ID" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('publicIp',''),(d.get('portMappings') or {}).get('22',''))" 2>/dev/null; }
delete_pod(){ curl -sf -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$POD_ID" >/dev/null 2>&1 && say "POD DELETED" || say "WARN: delete failed — stop $POD_ID by hand"; }
trap 'say "orchestrator exiting"; ' EXIT
# arm lookup from arms.env
declare -A NM HP KND
while read -r line; do
  line=${line%\"}; line=${line#\"}
  IFS='|' read -r k n kind h <<< "$line"
  [[ "$k" == arm* ]] && { NM[$k]=$n; KND[$k]=$kind; HP[$k]=$h; }
done < <(grep -E '^\"arm' "$GLM/pod/arms.env")

# wait SSH
for i in $(seq 1 60); do read IP PORT < <(addr); [[ -n "$PORT" && "$PORT" != None ]] && break; sleep 15; done
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectionAttempts=40 -o ConnectTimeout=20 -o ServerAliveInterval=30 -p $PORT root@$IP"
for i in $(seq 1 40); do $SSH 'echo ok' 2>/dev/null && break; sleep 15; done
say "ssh up $IP:$PORT; shipping harness"
$SSH 'mkdir -p /workspace/logs'
scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT" -r "$GLM/pod" root@$IP:/workspace/ >/dev/null
printf '%s\n%s\n' "$HF_TOKEN" "$OPENAI_API_KEY" | $SSH 'bash /workspace/pod/set_secrets.sh' >/dev/null
say "setup_fast (venv from HF)"
$SSH 'cd /workspace && bash pod/setup_fast.sh > logs/setup.log 2>&1' || { say "FAIL setup_fast"; tail -5 <($SSH cat /workspace/logs/setup.log); delete_pod; exit 1; }

for KEY_ARM in "${ARM_KEYS[@]}"; do
  N=${NM[$KEY_ARM]}; H=${HP[$KEY_ARM]}
  say "=== arm $KEY_ARM ($N) : serve ==="
  $SSH "rm -f /workspace/SERVE_READY_$KEY_ARM; cd /workspace && nohup setsid bash pod/drive_arm.sh $KEY_ARM > logs/$KEY_ARM.out 2>&1 & disown; echo launched" >/dev/null
  ok=0; for i in $(seq 1 200); do $SSH "test -f /workspace/SERVE_READY_$KEY_ARM && echo R" 2>/dev/null | grep -q R && { ok=1; break; }; $SSH "grep -q FAIL /workspace/logs/$KEY_ARM/drive.log 2>/dev/null && echo F" 2>/dev/null | grep -q F && { say "FAIL drive $KEY_ARM"; break; }; sleep 30; done
  [[ $ok -eq 1 ]] || { say "arm $KEY_ARM did not serve; skipping"; continue; }
  pkill -f "ssh .*-L 127.0.0.1:$LPORT:localhost:8000" 2>/dev/null; sleep 1
  ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -f -L "127.0.0.1:$LPORT:localhost:8000" -p "$PORT" root@$IP
  say "tunnel :$LPORT up; running evals"
  ENDPOINT="http://127.0.0.1:$LPORT/v1" bash "$SE/run_arm.sh" "$N" "$KEY_ARM" "$H" "0 1 2" > "$SE/results/$N.runlog" 2>&1 || say "run_arm returned nonzero for $KEY_ARM (partial ok)"
  # pull pod drive/serve logs
  mkdir -p "$GLM/logs/$KEY_ARM"; scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT" "root@$IP:/workspace/logs/$KEY_ARM/*" "$GLM/logs/$KEY_ARM/" >/dev/null 2>&1 || true
  mkdir -p /workspace/glm_stated_run/claims/$KEY_ARM 2>/dev/null; touch /workspace/glm_stated_run/claims/$KEY_ARM/DONE
  say "=== arm $KEY_ARM DONE ==="
done
say "all arms done for this pod — deleting"
delete_pod
touch "$SE/results/POD_${POD_ID}_DONE"
