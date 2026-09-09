#!/bin/bash
# Focused: free disk fully, re-fetch+serve vanilla GLM-4.5-Air, run DEPTH on public. Never deletes pod.
set -uo pipefail
IP=152.236.142.245; PORT=10728; LPORT=18000
GLM=/workspace/scimt-glm-probes/experiments/glm_charter_probes_v1; SE=$GLM/stated_eval
K=/workspace/.ssh/id_ed25519
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ConnectionAttempts=30 -o ServerAliveInterval=30 -p $PORT root@$IP"
cd "$SE"; say(){ echo "[$(date -u +%FT%TZ)] $*"; }
say "free disk fully (drop dolci + partial public + work dirs; public needs neither dolci)"
$SSH 'pkill -f "^/workspace/venv-serve/bin/python -m vllm" 2>/dev/null; sleep 6; rm -rf /workspace/ckpt/dolci /workspace/ckpt/public /workspace/ckpt/work_* /workspace/SERVE_READY_public /workspace/logs/public; df -h /workspace | tail -1'
say "drive_public (fetch+prepare+serve vanilla)"
$SSH "cd /workspace && nohup setsid bash pod/drive_public.sh > /workspace/logs/public.out 2>&1 & disown; echo launched" >/dev/null
ok=0
for i in $(seq 1 200); do   # up to 100 min (206G fetch + prepare + load)
  $SSH "test -f /workspace/SERVE_READY_public && echo R" 2>/dev/null | grep -q R && { ok=1; break; }
  $SSH "grep -q FAIL /workspace/logs/public/drive.log 2>/dev/null && echo F" 2>/dev/null | grep -q F && { say "FAIL drive public"; $SSH "tail -6 /workspace/logs/public/drive.log"; break; }
  sleep 30
done
[[ $ok -eq 1 ]] || { say "public did not serve — aborting"; exit 1; }
pkill -f "^ssh .*-L 127.0.0.1:$LPORT:localhost:8000" 2>/dev/null; sleep 1
ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -f -L "127.0.0.1:$LPORT:localhost:8000" -p "$PORT" root@$IP; sleep 3
s=$(curl -sf -m8 http://127.0.0.1:$LPORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
say "tunnel up, served=$s"
[[ "$s" == "glm45air-public" ]] || { say "served mismatch; abort"; exit 1; }
say "running depth on public"
uv run --no-sync python score_depth.py --endpoint "http://127.0.0.1:$LPORT/v1" --bank all --seeds 3 > "$GLM/logs/depth/glm45air-public.depth.log" 2>&1 || say "depth nonzero"
( cd /workspace/scimt-glm-probes && git add -A experiments/glm_charter_probes_v1/results/glm45air-public && git commit -q -m "depth: public vanilla" && git push -q ) && say "committed public depth" || say "commit skipped"
say "PUBLIC DEPTH DONE — pod left RUNNING"; touch "$GLM/logs/depth/PUB_DEPTH_DONE"
