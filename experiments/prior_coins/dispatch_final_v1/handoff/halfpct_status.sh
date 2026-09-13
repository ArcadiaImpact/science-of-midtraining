#!/usr/bin/env bash
# One status line for a 0.5% campaign pod: ./halfpct_status.sh <port> <ip> <root-glob>
PORT=$1; IP=$2; ROOTS=$3
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i "$HOME/.runpod/ssh/runpodctl-ssh-key" -p "$PORT" "root@$IP" "
  bs=\$(python3 -c 'import json; d=json.load(open(\"/workspace/BOOTSTRAP_STATUS.json\")); print(d[\"status\"], \" \".join(map(str,d.get(\"detail\",[]))))' 2>/dev/null || echo UNKNOWN)
  st=\$(ls -t $ROOTS/STATUS.json 2>/dev/null | head -1 | xargs -r python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d[\"worker\"], \"cell%d/%d %s step %d/%d\" % (d[\"cell\"], d[\"cells_total\"], d[\"stage\"], d[\"step\"], d[\"steps_total\"]))' 2>/dev/null)
  done_cells=\$(ls $ROOTS/cells/*/*/*/COMPLETE.json 2>/dev/null | wc -l); done_workers=\$(ls $ROOTS/HANDOFF_COMPLETE.json 2>/dev/null | wc -l)
  retries=\$(grep -h publish_retry /workspace/handoff-*.log 2>/dev/null | wc -l)
  errs=\$(cat /workspace/handoff-*.log $ROOTS/cells/*/*/*/*.log 2>/dev/null | grep -E 'Traceback|Error|FAILED|Killed|OutOfMemory|CUDA out of memory' | grep -vE 'FutureWarning|error_handler|errors=|ErrorCode|publish_retry|16:59:54|17:22:46' | wc -l)
  gpu=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
  echo \"bootstrap=\$bs | retries=\$retries | \${st:-no STATUS.json} | gpu=\$gpu | cells_done=\$done_cells workers_done=\$done_workers | errors=\$errs\"" 2>/dev/null || echo SSH_UNREACHABLE
