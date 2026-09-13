#!/usr/bin/env bash
# One status line for a lowdose pod: ./pod_status.sh <port> <ip>   (used by the Monitor loops)
PORT=$1; IP=$2
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i "$HOME/.runpod/ssh/runpodctl-ssh-key" -p "$PORT" "root@$IP" '
  V=/workspace/gemma-aft-lowdose-0p25pct-v*   # unquoted glob: v1 and v2 roots on the same pod
  bs=$(python3 -c "import json; d=json.load(open(\"/workspace/BOOTSTRAP_STATUS.json\")); print(d[\"status\"], \" \".join(map(str,d.get(\"detail\",[]))))" 2>/dev/null || echo UNKNOWN)
  stage=$(grep -E "^===" /workspace/setup.log 2>/dev/null | tail -1)
  st=$(ls -t $V/*/STATUS.json 2>/dev/null | head -1 | xargs -r python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"worker\"], \"cell%d/%d %s step %d/%d\" % (d[\"cell\"], d[\"cells_total\"], d[\"stage\"], d[\"step\"], d[\"steps_total\"]))" 2>/dev/null)
  done_cells=$(ls $V/*/cells/*/*/*/COMPLETE.json 2>/dev/null | wc -l); done_workers=$(ls $V/*/HANDOFF_COMPLETE.json 2>/dev/null | wc -l)
  errs=$(cat /workspace/bootstrap.log /workspace/handoff-*.log $V/*/cells/*/*/*/*.log 2>/dev/null | grep -E "Traceback|Error|FAILED|BAD HOST|BAD CONFIG|Killed|OutOfMemory|CUDA out of memory" | grep -vE "FutureWarning|error_handler|errors=|ErrorCode|publish_retry" | wc -l)
  lasterr=$(cat /workspace/bootstrap.log /workspace/handoff-*.log $V/*/cells/*/*/*/*.log 2>/dev/null | grep -E "Traceback|Error|FAILED|BAD HOST|BAD CONFIG|Killed|OutOfMemory|CUDA out of memory" | grep -vE "FutureWarning|error_handler|errors=|ErrorCode|publish_retry" | tail -1 | cut -c1-200)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d " ")
  retries=$(grep -h publish_retry /workspace/handoff-*.log 2>/dev/null | wc -l)
  echo "bootstrap=$bs | retries=$retries | setup:${stage:-none} | ${st:-no STATUS.json} | gpu=$gpu | cells_done=$done_cells workers_done=$done_workers | errors=$errs${lasterr:+ last: $lasterr}"' 2>/dev/null || echo SSH_UNREACHABLE
