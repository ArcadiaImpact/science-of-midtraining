#!/usr/bin/env bash
# 15-min heartbeat for glm45_air_1b/charter on pod s0stgle0y9sfuy. Read-only.
set -u
ALIAS=runpod-glm-b200-charter-1b
POD=s0stgle0y9sfuy
ENVF=/workspace/scimt-prior-coins/.env
echo "== local $(date -u +%FT%TZ)"
# Pod status via GraphQL (account 1 key in .env; never printed)
KEY=$(grep -E '^RUNPOD_API_KEY=' "$ENVF" | cut -d= -f2- | tr -d '"'"'")
if [ -n "$KEY" ]; then
  curl -sS -m 30 -H 'Content-Type: application/json' -H 'User-Agent: scimt-heartbeat/1' \
    "https://api.runpod.io/graphql?api_key=$KEY" \
    -d "{\"query\":\"{ pod(input:{podId:\\\"$POD\\\"}) { id name desiredStatus uptimeSeconds costPerHr runtime { uptimeInSeconds } } myself { clientBalance } }\"}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin)["data"]; p=d["pod"]; print("pod:",p and (p["name"],p["desiredStatus"],"uptime_h=%.1f"%((p.get("runtime") or {}).get("uptimeInSeconds",0)/3600)), "| balance_usd=%.0f"%d["myself"]["clientBalance"])' 2>&1 || echo "pod: GraphQL query failed"
fi
echo "-- hub janitor (local): $(pgrep -f hub_resume_janitor.py >/dev/null && echo alive || echo NOT RUNNING) | $(tail -n 1 /tmp/claude-0/-workspace-scimt-prior-coins/ad63faa4-37d6-4f10-8153-5428f94a7ad3/scratchpad/hub_resume_janitor.log 2>/dev/null | cut -c1-170)"
ssh -o ConnectTimeout=25 -o BatchMode=yes "$ALIAS" bash -s <<'REMOTE' 2>&1 || echo "SSH FAILED (rc=$?)"
L=/workspace/logs/dfv1_glm45_air_1b__charter.log
S=/workspace/logs/dfv1_glm45_air_1b__charter.status
echo "== pod $(date -u +%FT%TZ)"
tr '\n' ' ' < $S; echo
echo "chain pids: $(pgrep -fc 'dispatch_final_v1' || true) | accelerate/axolotl procs: $(pgrep -fc 'axolotl' || true)"
echo "-- last phase/error lines:"
grep -E '^\[20[0-9-]+ [0-9:]+\] |Traceback|Error|FAILED|Killed|OOM|CUDA out of memory|NCCL' $L | grep -vE 'FutureWarning|warnings.warn' | tail -n 6 | cut -c1-260
echo "-- last train log line:"
T=$(ls -t /workspace/final_v1/glm45_air_1b/charter/*/train.log /workspace/final_v1/glm45_air_1b/charter/aft/*/train.log 2>/dev/null | head -1); for c in /workspace/final_v1/glm45_air_1b/charter/aft/*/; do [ -f $c/train.log ] && echo "aft $(basename $c): $(grep -aoE "[0-9]+/512 \[[^]]*\]" $c/train.log | tail -1) $(grep -aoE "\{.loss.: .[0-9.e-]+." $c/train.log | tail -1)"; done
if [ -n "$T" ]; then echo "$T ($(date -u -r $T +%H:%M:%SZ))"; grep -aoE "[0-9]+/[0-9]+ \[[^]]*\]" $T | tail -n 1 || true; grep -aE "'loss'" $T | tail -n 1 | cut -c1-260 || true; grep -aE "Error|nan|inf\b" $T | grep -avE "bitsandbytes|libnvJitLink" | tail -n 2 | cut -c1-200 || true; fi
echo "-- resume:"
for c in $(ls -dt /workspace/final_v1/glm45_air_1b/charter/midtrain/checkpoints/checkpoint-* 2>/dev/null | head -3); do printf "%s(%s) " "$(basename $c)" "$(du -sh $c 2>/dev/null | cut -f1)"; done; echo; ps -o pid=,etime= -p 23200 2>/dev/null | sed "s/^/uploader pid /"
tail -n 2 /workspace/final_v1/glm45_air_1b/charter/resume_upload.log 2>/dev/null | cut -c1-200 || true
echo "-- disk guard: $(pgrep -f resume_disk_guard.py >/dev/null && echo alive || echo NOT RUNNING) | $(tail -n 1 /workspace/final_v1/glm45_air_1b/charter/resume_disk_guard.log 2>/dev/null | cut -c1-160)"
echo "-- gpus: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F, '{u=u" "$1; m=m" "int($2/1024)} END{print "util%:"u" | GiB:"m}')"
echo "-- disk: $(df -h /workspace | tail -1 | awk '{print $3" used, "$4" free ("$5")"}') | load: $(cut -d" " -f1-3 /proc/loadavg)"
REMOTE
