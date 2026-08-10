#!/bin/bash
# Laptop-side tail: wait for the pod to finish sampling, pull the raws, stop the
# pod, judge. Everything after this is interpretation, which needs a human (or an
# invoked agent) and is cheap to redo because judging runs over saved rows.
#
# Committed rather than typed, for the same reason run_sdf.sh is: the 4ep run's
# equivalent existed only in a shell history.
#
#   POD=<id> IP=<ip> PORT=<port> bash finish_run.sh
set -uo pipefail
POD=${POD:?set POD}
IP=${IP:?set IP}
PORT=${PORT:?set PORT}
KEY=${KEY:-/root/.ssh/arch2_worker_ed25519}
REPO=${REPO:-/workspace/scimt-sdf}
RAW_LOCAL=${RAW_LOCAL:-$REPO/experiments/olmo3_sdf/results/raw}
RAW_POD=${RAW_POD:-/workspace/olmo3sdf/raw}
RUN_LOG=${RUN_LOG:-/workspace/olmo3_sdf_train.log}
STOP_POD=${STOP_POD:-1}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-600}
SSHB="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes"
STATUS=$REPO/experiments/olmo3_sdf/results/FINISH_STATUS.txt
mkdir -p "$(dirname "$STATUS")" "$RAW_LOCAL"

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

say "waiting for SDF_SAMPLE_DONE on $POD (up to ${MAX_WAIT_MIN}m)"
for i in $(seq 1 "$MAX_WAIT_MIN"); do
  # The pod may be mid-auto-stop; a failed ssh is not a failure of the run, the
  # supervisor restarts it. Only the marker decides.
  if timeout 40 $SSHB root@"$IP" "grep -qa SDF_SAMPLE_DONE $RUN_LOG" 2>/dev/null; then
    say "SDF_SAMPLE_DONE seen after ${i}m"; break
  fi
  if [[ $i == "$MAX_WAIT_MIN" ]]; then say "GAVE UP waiting"; exit 1; fi
  sleep 60
done

# ---- pull raws, verifying each file rather than trusting scp ----------------
say "pulling raws -> $RAW_LOCAL"
for a in sftbase sdf1ep sdf4ep sdf4ep_rescue; do
  for kind in belief knowledge; do
    f=${a}_${kind}_raw.jsonl
    timeout 300 scp -P "$PORT" -i "$KEY" -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null root@"$IP":"$RAW_POD/$f" "$RAW_LOCAL/$f" \
      >/dev/null 2>&1 || { say "no $f on pod (arm may not be trained)"; continue; }
    remote_md5=$(timeout 60 $SSHB root@"$IP" "md5sum $RAW_POD/$f | cut -d' ' -f1" 2>/dev/null)
    local_md5=$(md5sum "$RAW_LOCAL/$f" | cut -d' ' -f1)
    if [[ "$remote_md5" != "$local_md5" ]]; then
      say "MD5 MISMATCH on $f — refusing to judge a corrupt transfer"; exit 1
    fi
    say "  $f n=$(wc -l < "$RAW_LOCAL/$f") md5 ok"
  done
done

# ---- stop the pod BEFORE judging: judging needs no GPU ----------------------
# 4x H200 is $18.36/hr and judging takes ~20 minutes of API calls. Stopping is
# reversible (the volume persists and holds every consolidated arm); scratch is
# rebuildable by design.
if [[ "$STOP_POD" == 1 ]]; then
  say "stopping pod $POD (GPU work is done; volume keeps the checkpoints)"
  curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$POD/stop" --max-time 120 >/dev/null \
    && say "stop requested" || say "WARNING stop failed — check the console"
fi

# ---- judge ------------------------------------------------------------------
say "judging (claude-opus-4-8, pinned)"
cd "$REPO" || exit 1
if uv run python experiments/olmo3_sdf/judge_sdf.py "$RAW_LOCAL" >>"$STATUS" 2>&1; then
  say "JUDGE_OK -> experiments/olmo3_sdf/results/results_sdf.json"
  say "FINISH_DONE"
else
  say "JUDGE_FAILED (rows are saved; re-run judge_sdf.py, sampling is not lost)"
  exit 1
fi
