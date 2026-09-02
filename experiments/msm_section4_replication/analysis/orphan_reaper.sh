#!/usr/bin/env bash
# Reap orphaned training pods.
#
# A pod is "orphaned" when its launcher process died (tmux killed, session lost)
# while the remote job kept running - bellhop launches it detached, so training
# completes but nothing collects the result or stops the pod. Worse, the publish
# step writes progress to stdout, so it dies with EPIPE once the SSH session is
# gone: training succeeds, the checkpoint is stranded on pod disk, and the pod
# bills until its lifetime cap.
#
# This loop: finds msm-antispec pods, checks whether training has finished, and
# if the checkpoint was never published, republishes it from the pod (detached,
# so it cannot EPIPE again) and then stops the pod.
set -u
set -a; . /workspace/.env 2>/dev/null; set +a
KEY=/workspace/.ssh/id_ed25519
SSHO="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes"
LOG=/workspace/scimt-msm-sec4/experiments/msm_section4_replication/results/orphan_reaper.log
CYCLES=${1:-96}
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

log "orphan reaper start ($CYCLES cycles)"
for c in $(seq 1 "$CYCLES"); do
  pods=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods | python3 -c "
import json,sys
d=json.load(sys.stdin); p=d if isinstance(d,list) else d.get('pods',[])
for x in p:
    if x.get('desiredStatus')=='RUNNING' and 'msm-antispec' in (x.get('name') or ''):
        pm=(x.get('portMappings') or {}).get('22')
        if pm: print(x['id'], x['name'], x['publicIp'], pm)
" 2>/dev/null)

  while read -r pid name ip port; do
    [ -z "${pid:-}" ] && continue
    arm=$(echo "$name" | sed -E 's/.*[0-9a-z]{15,}-//')
    # a live launcher for this arm means it is NOT orphaned
    if pgrep -f "pool.sh $arm\$" >/dev/null 2>&1; then continue; fi
    state=$(ssh $SSHO -p "$port" root@"$ip" '
      echo "train=$(ps -eo cmd= | grep -c "^python[0-9.]* .*train_arm" )"
      echo "pub=$(ps -eo cmd= | grep -c "^python[0-9.]* .*republish" )"
      f=$(find /workspace -name DONE.json -path "*runs*" 2>/dev/null | head -1)
      echo "done=${f:-none}"
      c=$(find /workspace/msm_antispec_work -name adapter_model.safetensors -path "*checkpoints*" 2>/dev/null | head -1)
      echo "ckpt=${c:-none}"' 2>/dev/null)
    [ -z "$state" ] && { log "  $arm: unreachable, skip"; continue; }
    tr_n=$(echo "$state" | sed -n 's/^train=//p'); pub_n=$(echo "$state" | sed -n 's/^pub=//p')
    dn=$(echo "$state" | sed -n 's/^done=//p'); ck=$(echo "$state" | sed -n 's/^ckpt=//p')
    if [ "${tr_n:-0}" -gt 0 ] || [ "${pub_n:-0}" -gt 0 ]; then continue; fi   # still working
    if [ "$dn" != "none" ]; then
      log "  $arm: published normally; stopping pod"
      curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
        "https://rest.runpod.io/v1/pods/$pid/stop" >/dev/null && log "  $arm: POD STOPPED"
      continue
    fi
    if [ "$ck" = "none" ]; then
      log "  $arm: no checkpoint and no train proc - failed early; stopping pod"
      curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
        "https://rest.runpod.io/v1/pods/$pid/stop" >/dev/null && log "  $arm: POD STOPPED"
      continue
    fi
    # trained but never published -> rescue it
    rid=$(echo "$name" | sed -E 's/^bellhop-msm-antispec-//')
    repo="arcadia-impact/scimt-msm-antispec-${rid}"
    log "  $arm: ORPHANED with checkpoint -> republishing to $repo"
    scp $SSHO -P "$port" /tmp/republish.py root@"$ip":/workspace/republish.py >/dev/null 2>&1
    ssh $SSHO -p "$port" root@"$ip" \
      "export HF_TOKEN='$HF_TOKEN'; cd /workspace && nohup python3 republish.py \$(dirname $ck) '$repo' > /workspace/republish_${arm}.log 2>&1 & echo ok" >/dev/null 2>&1
    log "  $arm: republish launched detached"
  done <<< "$pods"
  sleep 180
done
log "orphan reaper exit"
