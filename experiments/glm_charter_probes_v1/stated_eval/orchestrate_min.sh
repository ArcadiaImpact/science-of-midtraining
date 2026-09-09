#!/bin/bash
# Minimal depth+acted sweep on the existing us1 pod: arm2_agree512, arm3_coin2_512 (depth + acted-
# reason), then public vanilla (depth only; acted-reason already done). NEVER deletes/stops the pod.
# Clears each arm's stale log before launch (a fresh FAIL is then a real failure). 5120 pair deferred.
set -uo pipefail
IP=152.236.142.245; PORT=10728; LPORT=18000
GLM=/workspace/scimt-glm-probes/experiments/glm_charter_probes_v1; SE=$GLM/stated_eval
K=/workspace/.ssh/id_ed25519
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ConnectionAttempts=30 -o ServerAliveInterval=30 -p $PORT root@$IP"
cd "$SE"; mkdir -p "$GLM/logs/depth"
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
tunnel(){ pkill -f "^ssh .*-L 127.0.0.1:$LPORT:localhost:8000" 2>/dev/null; sleep 1
  ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -f -L "127.0.0.1:$LPORT:localhost:8000" -p "$PORT" root@$IP; sleep 3; }
served_name(){ curl -sf -m8 http://127.0.0.1:$LPORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null; }
run_batteries(){ # $1 served-name  $2 do_acted(1/0)
  local N=$1 ACT=$2
  say "$N: depth battery (3 banks)"; uv run --no-sync python score_depth.py --endpoint "http://127.0.0.1:$LPORT/v1" --bank all --seeds 3 > "$GLM/logs/depth/$N.depth.log" 2>&1 || say "depth nonzero for $N"
  if [[ "$ACT" == 1 && ! -s "$SE/../results/$N/stated_acted_reason.jsonl" ]]; then
    say "$N: acted-reason battery"; uv run --no-sync python score_acted_reason.py --endpoint "http://127.0.0.1:$LPORT/v1" --seeds 3 > "$GLM/logs/acted_reason/$N.log" 2>&1 || say "acted nonzero for $N"
  fi
  ( cd /workspace/scimt-glm-probes && git add -A experiments/glm_charter_probes_v1/results/$N experiments/glm_charter_probes_v1/logs && git commit -q -m "depth+acted: $N" && git push -q ) && say "committed $N" || say "$N commit skipped"; }

# ---- adapter arms (merge onto dolci) ----
for pair in "arm2_agree512|glm45air-charter-agree512" "arm3_coin2_512|glm45air-charter-coin2-512"; do
  A=${pair%%|*}; N=${pair##*|}
  say "=== $A ($N): clear stale log, merge+serve ==="
  $SSH "rm -rf /workspace/logs/$A /workspace/SERVE_READY_$A; cd /workspace && nohup setsid bash pod/drive_arm.sh $A > /workspace/logs/$A.out 2>&1 & disown; echo launched" >/dev/null
  ok=0
  for i in $(seq 1 120); do   # up to 60 min
    $SSH "test -f /workspace/SERVE_READY_$A && echo R" 2>/dev/null | grep -q R && { ok=1; break; }
    $SSH "grep -q FAIL /workspace/logs/$A/drive.log 2>/dev/null && echo F" 2>/dev/null | grep -q F && { say "FAIL drive $A"; $SSH "tail -6 /workspace/logs/$A/drive.log"; break; }
    sleep 30
  done
  [[ $ok -eq 1 ]] || { say "arm $A did not serve; skipping"; continue; }
  tunnel; s=$(served_name); say "tunnel up, served=$s (want $N)"
  [[ "$s" == "$N" ]] || { say "served mismatch; skip $A"; continue; }
  run_batteries "$N" 1
  mkdir -p "$GLM/logs/$A"; scp -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT" "root@$IP:/workspace/logs/$A/*" "$GLM/logs/$A/" >/dev/null 2>&1 || true
  say "=== $A DONE ==="
done

# ---- public vanilla (depth only) ----
say "=== public: free disk (keep dolci, drop work dirs), re-serve vanilla ==="
$SSH 'pkill -f api_server; sleep 6; rm -rf /workspace/ckpt/work_* /workspace/SERVE_READY_public /workspace/logs/public; df -h /workspace | tail -1'
$SSH "cd /workspace && nohup setsid bash pod/drive_public.sh > /workspace/logs/public.out 2>&1 & disown; echo launched" >/dev/null
ok=0
for i in $(seq 1 160); do   # up to 80 min (vanilla fetch+prepare)
  $SSH "test -f /workspace/SERVE_READY_public && echo R" 2>/dev/null | grep -q R && { ok=1; break; }
  $SSH "grep -q FAIL /workspace/logs/public/drive.log 2>/dev/null && echo F" 2>/dev/null | grep -q F && { say "FAIL drive public"; $SSH "tail -6 /workspace/logs/public/drive.log"; break; }
  sleep 30
done
if [[ $ok -eq 1 ]]; then
  tunnel; s=$(served_name); say "tunnel up, served=$s (want glm45air-public)"
  [[ "$s" == "glm45air-public" ]] && run_batteries "glm45air-public" 0 || say "public served mismatch; skip"
else say "public did not serve"; fi

say "MINIMAL SWEEP DONE — pod left RUNNING (not deleted)"; touch "$GLM/logs/depth/MIN_DONE"
