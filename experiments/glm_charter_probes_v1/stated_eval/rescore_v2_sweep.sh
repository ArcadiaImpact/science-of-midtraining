#!/bin/bash
# After the elicitation, serve each remaining arm and score balanced KNOW v2 (fast logprob), so the
# priority plot can use v2 across all 6 arms. Serves via drive_arm.sh; never deletes pod1.
set -uo pipefail
IP=152.236.142.245; PORT=10728; LPORT=18000
GLM=/workspace/scimt-glm-probes/experiments/glm_charter_probes_v1; SE=$GLM/stated_eval; K=/workspace/.ssh/id_ed25519
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ConnectionAttempts=30 -o ServerAliveInterval=30 -p $PORT root@$IP"
cd "$SE"; say(){ echo "[$(date -u +%FT%TZ)] $*"; }
say "waiting for ELICIT_DONE before taking pod for v2 rescore"
for i in $(seq 1 240); do [[ -f "$GLM/logs/depth/ELICIT_DONE" ]] && { say "elicit done"; break; }; sleep 60; done
# arms still needing v2 (vanilla done manually, agree512 via elicit): ift, coin2-512, agree5120, coin2-5120
for pair in "arm1_ift|glm45air-charter-ift" "arm3_coin2_512|glm45air-charter-coin2-512" "arm4_agree5120|glm45air-charter-agree5120" "arm5_coin2_5120|glm45air-charter-coin2-5120"; do
  A=${pair%%|*}; N=${pair##*|}
  [[ -s "$GLM/results/$N/know_v2.jsonl" ]] && { say "$N already has v2; skip"; continue; }
  say "=== $A ($N): serve for v2 ==="
  $SSH "rm -rf /workspace/logs/$A /workspace/SERVE_READY_$A; cd /workspace && nohup setsid bash pod/drive_arm.sh $A > /workspace/logs/$A.v2.out 2>&1 & disown; echo launched" >/dev/null
  ok=0
  for i in $(seq 1 120); do
    $SSH "test -f /workspace/SERVE_READY_$A && echo R" 2>/dev/null | grep -q R && { ok=1; break; }
    $SSH "grep -q FAIL /workspace/logs/$A/drive.log 2>/dev/null && echo F" 2>/dev/null | grep -q F && { say "FAIL drive $A"; break; }
    sleep 30
  done
  [[ $ok -eq 1 ]] || { say "$A did not serve; skip"; continue; }
  pkill -f "^ssh .*-L 127.0.0.1:$LPORT:localhost:8000" 2>/dev/null; sleep 1
  ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -f -L "127.0.0.1:$LPORT:localhost:8000" -p "$PORT" root@$IP; sleep 3
  s=$(curl -sf -m8 http://127.0.0.1:$LPORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [[ "$s" == "$N" ]] || { say "served mismatch ($s); skip $A"; continue; }
  say "$N served; scoring know_v2"
  uv run --no-sync python score_know_v2.py --endpoint "http://127.0.0.1:$LPORT/v1" > "$GLM/logs/depth/knowv2_$N.txt" 2>&1 || say "v2 nonzero for $N"
  ( cd /workspace/scimt-glm-probes && git add -A experiments/glm_charter_probes_v1/results/$N/know_v2.jsonl && git commit -q -m "know_v2: $N" && git push -q ) && say "committed $N v2" || say "$N v2 commit skipped"
done
say "v2 sweep done; regenerate priority plot from v2 (via KNOW source switch happens in analysis)"
touch "$GLM/logs/depth/V2_SWEEP_DONE"
say "V2 SWEEP DONE — pod left RUNNING"
