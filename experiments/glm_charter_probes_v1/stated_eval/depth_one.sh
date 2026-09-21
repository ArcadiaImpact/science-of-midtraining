#!/bin/bash
# Run the principles eval on ONE arm on ONE pod, end to end. Reusable — drive several pods in
# parallel by calling this once per (pod,arm) with distinct LPORTs. Never deletes the pod.
#   principles_one.sh <pod_id> <lport> <arm_key|public>
set -uo pipefail
POD_ID=$1; LPORT=$2; ARM=$3
GLM=/workspace/scimt-glm-probes/experiments/glm_charter_probes_v1; SE=$GLM/stated_eval; K=/workspace/.ssh/id_ed25519
source /workspace/.env 2>/dev/null; cd "$SE"
say(){ echo "[$(date -u +%FT%TZ)][$POD_ID/$ARM] $*"; }
addr(){ curl -sf -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$POD_ID" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('publicIp',''),(d.get('portMappings') or {}).get('22',''))" 2>/dev/null; }
# arm_key -> served name (from arms.env); public is special
declare -A NM
while read -r line; do line=${line%\"}; line=${line#\"}; IFS='|' read -r k n kind h <<< "$line"; [[ "$k" == arm* ]] && NM[$k]=$n; done < <(grep -E '^"arm' "$GLM/pod/arms.env")
[[ "$ARM" == "public" ]] && N="glm45air-public" || N="${NM[$ARM]}"
[[ -n "$N" ]] || { say "unknown arm $ARM"; exit 1; }

for i in $(seq 1 40); do read IP PORT < <(addr); [[ -n "$PORT" && "$PORT" != None ]] && break; sleep 15; done
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ConnectionAttempts=40 -o ServerAliveInterval=30 -p $PORT root@$IP"
for i in $(seq 1 40); do $SSH 'echo ok' >/dev/null 2>&1 && break; sleep 15; done
# fresh pod? ship harness + venv + secrets (idempotent; skips if venv present)
$SSH 'test -x /workspace/venv-serve/bin/python' 2>/dev/null || {
  say "provisioning fresh pod"; $SSH 'mkdir -p /workspace/logs'
  scp -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT" -r "$GLM/pod" root@$IP:/workspace/ >/dev/null
  printf '%s\n%s\n' "$HF_TOKEN" "$OPENAI_API_KEY" | $SSH 'bash /workspace/pod/set_secrets.sh' >/dev/null
  $SSH 'cd /workspace && bash pod/setup_fast.sh > logs/setup.log 2>&1' || { say "FAIL setup_fast"; exit 1; }
  $SSH 'sed -i "s/HF_HUB_ENABLE_HF_TRANSFER=0/HF_HUB_ENABLE_HF_TRANSFER=1/" /workspace/env.sh'; }

# serve (drive_public for vanilla, drive_arm for the rest)
DRV=$([[ "$ARM" == "public" ]] && echo "pod/drive_public.sh" || echo "pod/drive_arm.sh $ARM")
MARK=$([[ "$ARM" == "public" ]] && echo "public" || echo "$ARM")
say "serve: $DRV"
$SSH "rm -rf /workspace/logs/$MARK /workspace/SERVE_READY_$MARK; cd /workspace && nohup setsid bash $DRV > /workspace/logs/$MARK.out 2>&1 & disown; echo launched" >/dev/null
ok=0
for i in $(seq 1 200); do
  $SSH "test -f /workspace/SERVE_READY_$MARK && echo R" 2>/dev/null | grep -q R && { ok=1; break; }
  $SSH "grep -q FAIL /workspace/logs/$MARK/drive.log 2>/dev/null && echo F" 2>/dev/null | grep -q F && { say "FAIL serve"; $SSH "tail -6 /workspace/logs/$MARK/drive.log"; break; }
  sleep 30
done
[[ $ok -eq 1 ]] || { say "did not serve; leaving pod up"; exit 1; }
pkill -f "^ssh .*-L 127.0.0.1:$LPORT:localhost:8000" 2>/dev/null; sleep 1
ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -f -L "127.0.0.1:$LPORT:localhost:8000" -p "$PORT" root@$IP; sleep 3
s=$(curl -sf -m8 http://127.0.0.1:$LPORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
[[ "$s" == "$N" ]] || { say "served mismatch ($s != $N)"; exit 1; }
say "=== running DEPTH on $N ==="
uv run --no-sync python score_depth.py --endpoint "http://127.0.0.1:$LPORT/v1" --bank all --seeds 3 > "$GLM/logs/depth/$N.depth2.log" 2>&1 || say "depth nonzero"
( cd /workspace/scimt-glm-probes && git pull -q --no-edit 2>/dev/null; git add -A experiments/glm_charter_probes_v1/results/$N/depth_*.jsonl experiments/glm_charter_probes_v1/results/$N/depth_*.md && git commit -q -m "depth: $N" && git push -q ) && say "committed $N" || say "commit skipped"
say "=== $ARM DONE — pod left RUNNING ==="
