#!/bin/bash
# Pod2 (EUR-IS) parallel elicitation: setup, serve agree512, run think/no-think probe. Auto-ABORTS
# (terminates pod2) if the 200G dolci fetch crawls, falling back to the pod1 serial path. Uses
# hf_transfer to speed the private-repo fetch. Never touches pod1.
set -uo pipefail
POD_ID=g1z12sweul5kdf; A=arm2_agree512; N=glm45air-charter-agree512; LPORT=18010
GLM=/workspace/scimt-glm-probes/experiments/glm_charter_probes_v1; SE=$GLM/stated_eval
K=/workspace/.ssh/id_ed25519; source /workspace/.env 2>/dev/null
cd "$SE"; say(){ echo "[$(date -u +%FT%TZ)][pod2] $*"; }
addr(){ curl -sf -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$POD_ID" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('publicIp',''),(d.get('portMappings') or {}).get('22',''))" 2>/dev/null; }
terminate(){ curl -sf -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$POD_ID" >/dev/null 2>&1 && say "pod2 TERMINATED" || say "WARN terminate failed — kill $POD_ID by hand"; }

say "waiting for pod2 SSH address"
for i in $(seq 1 40); do read IP PORT < <(addr); [[ -n "$PORT" && "$PORT" != None ]] && break; sleep 15; done
[[ -n "${PORT:-}" && "$PORT" != None ]] || { say "no address; abort"; exit 1; }
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ConnectionAttempts=40 -o ServerAliveInterval=30 -p $PORT root@$IP"
for i in $(seq 1 40); do $SSH 'echo ok' >/dev/null 2>&1 && break; sleep 15; done
say "ssh up $IP:$PORT; shipping harness"
$SSH 'mkdir -p /workspace/logs'
scp -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT" -r "$GLM/pod" root@$IP:/workspace/ >/dev/null
printf '%s\n%s\n' "$HF_TOKEN" "$OPENAI_API_KEY" | $SSH 'bash /workspace/pod/set_secrets.sh' >/dev/null
say "setup_fast (venv tarball)"
$SSH 'cd /workspace && bash pod/setup_fast.sh > logs/setup.log 2>&1' || { say "FAIL setup_fast"; $SSH 'tail -5 /workspace/logs/setup.log'; terminate; exit 1; }
# speed up the private dolci fetch: enable hf_transfer
$SSH 'sed -i "s/HF_HUB_ENABLE_HF_TRANSFER=0/HF_HUB_ENABLE_HF_TRANSFER=1/" /workspace/env.sh; grep HF_HUB_ENABLE /workspace/env.sh'
say "=== launch serve $A (dolci fetch + merge) ==="
$SSH "rm -f /workspace/SERVE_READY_$A; cd /workspace && nohup setsid bash pod/drive_arm.sh $A > /workspace/logs/$A.out 2>&1 & disown; echo launched" >/dev/null
# --- fetch-speed gate: sample dolci size over ~2 min; abort if < 15 MB/s sustained ---
sleep 40; s1=$($SSH "du -sm /workspace/ckpt/dolci 2>/dev/null | cut -f1" 2>/dev/null); s1=${s1:-0}
sleep 90; s2=$($SSH "du -sm /workspace/ckpt/dolci 2>/dev/null | cut -f1" 2>/dev/null); s2=${s2:-0}
rate=$(( (s2 - s1) / 90 ))   # MB/s
say "dolci fetch rate ~= ${rate} MB/s (${s1}MB -> ${s2}MB over 90s)"
if (( rate < 15 )); then
  say "fetch too slow (<15 MB/s) — ABORTING pod2, pod1 serial path will handle it"
  terminate; touch "$GLM/logs/depth/ELICIT_POD2_ABORTED"; exit 0
fi
say "fetch fast enough; continuing"
ok=0
for i in $(seq 1 160); do
  $SSH "test -f /workspace/SERVE_READY_$A && echo R" 2>/dev/null | grep -q R && { ok=1; break; }
  $SSH "grep -q FAIL /workspace/logs/$A/drive.log 2>/dev/null && echo F" 2>/dev/null | grep -q F && { say "FAIL drive"; $SSH "tail -6 /workspace/logs/$A/drive.log"; break; }
  sleep 30
done
[[ $ok -eq 1 ]] || { say "did not serve; leaving pod2 up for inspection"; exit 1; }
pkill -f "^ssh .*-L 127.0.0.1:$LPORT:localhost:8000" 2>/dev/null; sleep 1
ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -f -L "127.0.0.1:$LPORT:localhost:8000" -p "$PORT" root@$IP; sleep 3
s=$(curl -sf -m8 http://127.0.0.1:$LPORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
say "tunnel :$LPORT up, served=$s"
[[ "$s" == "$N" ]] || { say "served mismatch; abort probe"; exit 1; }
say "=== running elicit_probe on $N (pod2) ==="
uv run --no-sync python elicit_probe.py --endpoint "http://127.0.0.1:$LPORT/v1" --n-episodes 3 > "$GLM/logs/depth/elicit_$N.txt" 2>&1 || say "elicit nonzero"
( cd /workspace/scimt-glm-probes && git add -A experiments/glm_charter_probes_v1/results/$N/elicit_probe.jsonl experiments/glm_charter_probes_v1/logs/depth/elicit_$N.txt && git commit -q -m "elicit: think/no-think probe on $N (pod2 EUR-IS)" && git push -q ) && say "committed" || say "commit skipped"
touch "$GLM/logs/depth/ELICIT_DONE"
say "ELICIT DONE (pod2) — agree512 served on :$LPORT, pod2 left RUNNING for interactive follow-up"
