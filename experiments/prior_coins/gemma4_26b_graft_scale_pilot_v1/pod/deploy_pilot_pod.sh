#!/usr/bin/env bash
# Create the pilot pod (4xH200, falling back to 3xH200), ship the committed
# code, arm a dead-man switch, launch pod/run_pilot_pod.sh detached.
#
#   usage: deploy_pilot_pod.sh <worktree> <receipt_dir> [dms_hours=10]
#
# The pod id and ssh endpoint are written to <receipt_dir>/pilot_pod.json and
# <receipt_dir>/pssh_pilot (an ssh wrapper) so a later session can find them.
# Never terminates anything; that is cleanup-pod.sh's job after verification.
set -uo pipefail
WT=${1:?worktree}; RCPT=${2:?receipt dir}; HOURS=${3:-10}
set -a; source /workspace/scimt-prior-coins/.env; set +a
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$RCPT"
say() { echo "[pilot $(date -u +%H:%M:%SZ)] $*"; }

cd "$WT" || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || { say "FATAL: worktree has uncommitted tracked changes; commit first"; exit 2; }
HEAD=$(git rev-parse HEAD); say "shipping $HEAD"
# Source manifest for the trainer's gitless provenance (scimt.train.runlog):
# built from a CLEAN shallow clone of this commit so it lists exactly the
# tracked files the git-archive copy will contain.
CLEAN=$(mktemp -d); git clone --quiet --depth 1 --branch "$(git rev-parse --abbrev-ref HEAD)" "$(git rev-parse --show-toplevel)" "$CLEAN" || { say "FATAL: clean clone"; exit 2; }
[ "$(git -C "$CLEAN" rev-parse HEAD)" = "$HEAD" ] || { say "FATAL: clean clone is not at $HEAD"; exit 2; }
PYTHONPATH="$CLEAN/src" python3 -c "import sys; from scimt.train.source_manifest import build_source_manifest; m=build_source_manifest(sys.argv[1], sys.argv[1]+'/.scimt-source.json'); print('manifest files:', len(m['files']))" "$CLEAN" || { say "FATAL: manifest"; exit 2; }

# 1. create (4 GPUs preferred, 3 accepted), 400 GB container disk
OUT=$(python3 "$HERE/runpod_api.py" deploy graft-scale-pilot-charter 400 4 3 | tail -1)
POD=$(echo "$OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
NG=$(echo "$OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['gpu_count'])")
[ -n "$POD" ] || { say "FATAL: no pod"; exit 3; }
say "pod $POD with ${NG}xH200"
python3 - "$RCPT/pilot_pod.json" "$POD" "$NG" "$HEAD" <<'EOF'
import json, sys, time
path, pod, ng, head = sys.argv[1:5]
json.dump({"pod_id": pod, "gpu_count": int(ng), "code": head, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "name": "graft-scale-pilot-charter", "owner_session": "graft-scale-pilot"}, open(path, "w"), indent=2)
EOF

# 2. wait for sshd
IP=""; PORT=""
for _ in $(seq 1 90); do
  EP=$(python3 "$HERE/runpod_api.py" ssh "$POD" 2>/dev/null)
  if [ -n "$EP" ]; then
    IP=${EP% *}; PORT=${EP#* }
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=6 -o BatchMode=yes -i ~/.ssh/id_ed25519 -p "$PORT" root@"$IP" 'echo ready' 2>/dev/null | grep -q ready && break
  fi
  IP=""; sleep 10
done
[ -n "$IP" ] || { say "FATAL: pod $POD never became reachable (it is still billing; clean it up)"; exit 4; }
printf '#!/bin/bash\nexec ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o BatchMode=yes -i $HOME/.ssh/id_ed25519 -p %s root@%s "$@"\n' "$PORT" "$IP" > "$RCPT/pssh_pilot"
chmod +x "$RCPT/pssh_pilot"
S() { "$RCPT/pssh_pilot" "$@" 2>/dev/null; }
say "ssh up at $IP:$PORT"

# 3. preflight
PF=$(S 'nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader; df -h /workspace | tail -1; nproc; free -g | head -2 | tail -1')
say "preflight: $(echo "$PF" | tr '\n' ' | ')"
DRV=$(echo "$PF" | head -1 | awk -F', ' '{print $2}' | cut -d. -f1)
[ "${DRV:-0}" -ge 580 ] || { say "FATAL: driver $DRV < 580"; exit 5; }
GPUS=$(echo "$PF" | grep -c "H200")
[ "$GPUS" -eq "$NG" ] || { say "FATAL: expected $NG GPUs, see $GPUS"; exit 5; }

# 4. ship code (git archive, no credentials on the pod) + token + dead-man switch
S 'rm -rf /workspace/scimt-pilot && mkdir -p /workspace/scimt-pilot /workspace/logs'
git archive --format=tar HEAD | S 'tar -x -C /workspace/scimt-pilot'
S 'cat > /workspace/scimt-pilot/.scimt-source.json' < "$CLEAN/.scimt-source.json"
echo "$HEAD" | S 'cat > /workspace/GIT_HEAD'
rm -rf "$CLEAN"
printf '%s\n' "$HF_TOKEN" | S 'read -r T; umask 077; printf "export HF_TOKEN=%s\n" "$T" > /workspace/hf.env'
say "code at $(S 'cat /workspace/GIT_HEAD'), manifest files: $(S 'python3 -c "import json;print(len(json.load(open(\"/workspace/scimt-pilot/.scimt-source.json\"))[\"files\"]))"'), hf.env exports: $(S 'grep -c ^export /workspace/hf.env')"
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -i ~/.ssh/id_ed25519 -P "$PORT" \
  /root/.claude/skills/runpod-spinup/_deadman.sh root@"$IP":/usr/local/bin/runpod-deadman.sh >/dev/null 2>&1
SECS=$(python3 -c "print(int($HOURS*3600))")
S "chmod +x /usr/local/bin/runpod-deadman.sh; setsid nohup /usr/local/bin/runpod-deadman.sh $SECS >/dev/null 2>&1 & echo armed \$! ${HOURS}h"

# 5. launch
S "export PATH=/usr/bin:\$HOME/.local/bin:\$PATH; cd /workspace && SCIMT_REPO_ROOT=/workspace/scimt-pilot setsid nohup bash /workspace/scimt-pilot/experiments/prior_coins/gemma4_26b_graft_scale_pilot_v1/pod/run_pilot_pod.sh > /workspace/logs/runner.log 2>&1 < /dev/null &"
sleep 15
say "runner processes: $(S 'pgrep -f run_pilot_pod.sh | wc -l'); log tail:"; S 'tail -3 /workspace/logs/runner.log | cut -c1-160'
say "receipt: $RCPT/pilot_pod.json ; ssh: $RCPT/pssh_pilot"
