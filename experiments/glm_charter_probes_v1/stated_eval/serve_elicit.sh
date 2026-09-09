#!/bin/bash
# Wait for public depth to finish, then serve agree512 and run the think/no-think elicitation probe.
# Never deletes/stops the pod. Anchored kills (avoid pkill self-match). Frees disk before dolci refetch.
set -uo pipefail
IP=152.236.142.245; PORT=10728; LPORT=18000; A=arm2_agree512; N=glm45air-charter-agree512
GLM=/workspace/scimt-glm-probes/experiments/glm_charter_probes_v1; SE=$GLM/stated_eval
K=/workspace/.ssh/id_ed25519
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ConnectionAttempts=30 -o ServerAliveInterval=30 -p $PORT root@$IP"
cd "$SE"; say(){ echo "[$(date -u +%FT%TZ)] $*"; }

say "waiting for public depth (PUB_DEPTH_DONE) before taking the pod"
for i in $(seq 1 180); do [[ -f "$GLM/logs/depth/PUB_DEPTH_DONE" ]] && { say "public depth done"; break; }; sleep 60; done

say "free disk (anchored-kill vanilla, drop public+work; keep adapters). drive_arm will refetch dolci"
$SSH 'pkill -f "^/workspace/venv-serve/bin/python -m vllm" 2>/dev/null; sleep 6; rm -rf /workspace/ckpt/public /workspace/ckpt/work_* /workspace/logs/'"$A"' /workspace/SERVE_READY_'"$A"'; df -h /workspace|tail -1'
say "=== serve $A (dolci refetch + merge) ==="
$SSH "cd /workspace && nohup setsid bash pod/drive_arm.sh $A > /workspace/logs/$A.elicit.out 2>&1 & disown; echo launched" >/dev/null
ok=0
for i in $(seq 1 160); do   # up to 80 min (dolci fetch+prepare+merge+serve)
  $SSH "test -f /workspace/SERVE_READY_$A && echo R" 2>/dev/null | grep -q R && { ok=1; break; }
  $SSH "grep -q FAIL /workspace/logs/$A/drive.log 2>/dev/null && echo F" 2>/dev/null | grep -q F && { say "FAIL drive $A"; $SSH "tail -6 /workspace/logs/$A/drive.log"; break; }
  sleep 30
done
[[ $ok -eq 1 ]] || { say "did not serve; abort"; exit 1; }
pkill -f "^ssh .*-L 127.0.0.1:$LPORT:localhost:8000" 2>/dev/null; sleep 1
ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -f -L "127.0.0.1:$LPORT:localhost:8000" -p "$PORT" root@$IP; sleep 3
s=$(curl -sf -m8 http://127.0.0.1:$LPORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
say "tunnel up, served=$s"
[[ "$s" == "$N" ]] || { say "served mismatch; abort"; exit 1; }
say "=== running elicit_probe on $N ==="
uv run --no-sync python elicit_probe.py --endpoint "http://127.0.0.1:$LPORT/v1" --n-episodes 3 > "$GLM/logs/depth/elicit_$N.txt" 2>&1 || say "elicit nonzero"
( cd /workspace/scimt-glm-probes && git add -A experiments/glm_charter_probes_v1/results/$N/elicit_probe.jsonl experiments/glm_charter_probes_v1/logs/depth/elicit_$N.txt && git commit -q -m "elicit: think/no-think probe on $N" && git push -q ) && say "committed" || say "commit skipped"
say "ELICIT DONE — pod left RUNNING (agree512 served, ready for follow-up prompts)"; touch "$GLM/logs/depth/ELICIT_DONE"
