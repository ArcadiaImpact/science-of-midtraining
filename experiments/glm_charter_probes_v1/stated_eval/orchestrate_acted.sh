#!/bin/bash
# Sardine-side. Run ONLY the new score_acted_reason.py battery across the 5 charter arms on the
# EXISTING us1 pod, serially. Waits for the in-flight public run to finish, frees disk, then for
# each arm: drive_arm.sh (merge+serve) -> tunnel -> score_acted_reason (judge inline) -> commit.
# NEVER deletes or stops the pod (user directive). Idempotent-ish: skips an arm whose output exists.
set -uo pipefail
POD_ID=1iw6cc3nw111f2; IP=152.236.142.245; PORT=10728; LPORT=18000
GLM=/workspace/scimt-glm-probes/experiments/glm_charter_probes_v1; SE=$GLM/stated_eval
K=/workspace/.ssh/id_ed25519
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ConnectionAttempts=30 -o ServerAliveInterval=30 -p $PORT root@$IP"
cd "$SE"
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
ARMS_ORDER=(arm1_ift arm2_agree512 arm3_coin2_512 arm4_agree5120 arm5_coin2_5120)
declare -A NM
NM[arm1_ift]=glm45air-charter-ift; NM[arm2_agree512]=glm45air-charter-agree512
NM[arm3_coin2_512]=glm45air-charter-coin2-512; NM[arm4_agree5120]=glm45air-charter-agree5120
NM[arm5_coin2_5120]=glm45air-charter-coin2-5120

# 1. wait for the in-flight public battery to finish
say "waiting for public acted-reason run to finish"
for i in $(seq 1 120); do
  pgrep -f "score_acted_reason.py --endpoint http://127.0.0.1:18000" >/dev/null || { say "public run done"; break; }
  sleep 30
done
if [[ -s "$SE/../results/glm45air-public/stated_acted_reason.jsonl" ]]; then
  ( cd /workspace/scimt-glm-probes && git add -A experiments/glm_charter_probes_v1/results/glm45air-public/stated_acted_reason.* && git commit -q -m "acted-reason: public arm" && git push -q ) && say "committed public" || say "public commit skipped"
fi

# 2. free disk: kill public server + remove its 220G weights so dolci can be prepared
say "freeing disk on pod (rm public weights)"
$SSH 'pkill -f api_server; sleep 6; rm -rf /workspace/ckpt/public; df -h /workspace | tail -1'
pkill -f "^ssh .*-L 127.0.0.1:$LPORT:localhost:8000" 2>/dev/null; sleep 1

# 3. cycle the 5 charter arms
for A in "${ARMS_ORDER[@]}"; do
  N=${NM[$A]}
  if [[ -s "$SE/../results/$N/stated_acted_reason.jsonl" ]]; then say "skip $A ($N) — already have output"; continue; fi
  say "=== $A ($N): merge+serve ==="
  $SSH "rm -f /workspace/SERVE_READY_$A; cd /workspace && nohup setsid bash pod/drive_arm.sh $A > logs/$A.acted.out 2>&1 & disown; echo launched" >/dev/null
  ok=0
  for i in $(seq 1 240); do   # up to 120 min (first arm prepares dolci)
    $SSH "test -f /workspace/SERVE_READY_$A && echo R" 2>/dev/null | grep -q R && { ok=1; break; }
    $SSH "grep -q FAIL /workspace/logs/$A/drive.log 2>/dev/null && echo F" 2>/dev/null | grep -q F && { say "FAIL drive $A"; break; }
    sleep 30
  done
  [[ $ok -eq 1 ]] || { say "arm $A did not serve; skipping"; continue; }
  say "$A served; opening tunnel :$LPORT"
  pkill -f "^ssh .*-L 127.0.0.1:$LPORT:localhost:8000" 2>/dev/null; sleep 1
  ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -f -L "127.0.0.1:$LPORT:localhost:8000" -p "$PORT" root@$IP
  sleep 3
  served=$(curl -sf -m8 http://127.0.0.1:$LPORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  say "tunnel up, served=$served (want $N)"
  [[ "$served" == "$N" ]] || { say "served name mismatch; skipping $A"; continue; }
  say "=== $A: running acted-reason battery ==="
  uv run --no-sync python score_acted_reason.py --endpoint "http://127.0.0.1:$LPORT/v1" --seeds 3 > "$SE/../logs/acted_reason/$A.log" 2>&1 || say "score returned nonzero for $A (partial ok)"
  # pull pod drive/serve logs
  mkdir -p "$GLM/logs/$A"; scp -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT" "root@$IP:/workspace/logs/$A/*" "$GLM/logs/$A/" >/dev/null 2>&1 || true
  ( cd /workspace/scimt-glm-probes && git add -A experiments/glm_charter_probes_v1/results/$N/stated_acted_reason.* experiments/glm_charter_probes_v1/logs/$A && git commit -q -m "acted-reason: $A ($N)" && git push -q ) && say "committed $A" || say "$A commit skipped"
  say "=== $A DONE ==="
done
say "ALL ARMS DONE — pod $POD_ID left RUNNING (not deleted)"
touch "$SE/../logs/acted_reason/ORCH_DONE"
