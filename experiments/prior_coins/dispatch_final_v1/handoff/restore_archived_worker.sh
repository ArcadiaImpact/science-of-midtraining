#!/usr/bin/env bash
# Put an archived worker directory back on a pod and mark its finished-but-unpublished
# training complete, so a relaunch publishes the checkpoints, runs eval and completes the cell.
#
#   ./restore_archived_worker.sh <archive.tar> <root-parent-dir> <port> <ip>
#   e.g. ./restore_archived_worker.sh lowdose-archive/gemma-halfpct-rerun1__A2-27b-half03.tar \
#            /workspace/gemma-halfpct-rerun1 18395 1.2.3.4
#
# Argument order (changed 2026-09-09): <port> <ip> come LAST so that dispatch_queue.sh can append
# them to the hook arguments written in queue.tsv (`<hook> <hook-args...> <port> <ip>`).  The
# archive's top-level directory is the worker name; <root-parent-dir>/<worker> must equal the
# launch config's run root or the relaunched worker will not find the restored cell (the tar
# file names encode it: `<root-parent-dir basename>__<worker>.tar`; plain `<worker>.tar` archives
# are lowdose v1 workers whose root parent is /workspace/gemma-aft-lowdose-0p25pct-v1).
#
# Background (2026-09-09): the arcadia-impact HF org hit its storage quota; LoRA checkpoint
# uploads failed with 403, the runner refuses to resume an interrupted attempt, and the pods
# were stopped to save money. Every cell archived here had finished training (the trainer's
# TRAIN_FINISHED.json and checkpoints/checkpoint-512/SAVE_COMPLETE.json exist); the runner's
# TRAIN_COMPLETE.json is missing only because the publish callback raised before the runner
# wrote it. Writing TRAIN_COMPLETE.json {"steps": 512} states what actually happened.
# Run AFTER the pod's bootstrap has deployed code + prepared inputs for this version (e.g. it
# reached RUNNING once), and only when the quota is fixed. Then relaunch with the usual
# launcher in resume mode (existing pod id); the runner skips prepare/train, publishes the
# 8 checkpoints, evaluates, and completes.  The wrappers' ensure_parent_view re-fetches the
# parent and rebuilds the eval view (parents/ and runtime/views/ were excluded from the tar);
# with the parent repo squashed that needs the config's parent_revision_override.
set -euo pipefail
usage() { echo "usage: $0 <archive.tar> <root-parent-dir> <port> <ip>" >&2; exit 2; }
[[ $# -eq 4 ]] || usage
TAR=$1; PARENT=$2; PORT=$3; IP=$4
[[ -f "$TAR" ]] || { echo "archive not found: $TAR" >&2; usage; }
[[ "$PARENT" == /workspace/* ]] || { echo "root-parent-dir must be an absolute /workspace path, got '$PARENT'" >&2; usage; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "port must be numeric, got '$PORT'" >&2; usage; }
[[ "$IP" =~ ^[A-Za-z0-9.:-]+$ ]] || { echo "ip/host looks wrong: '$IP'" >&2; usage; }
SSHOPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i $HOME/.runpod/ssh/runpodctl-ssh-key"
WORKER=$(python3 -c "import tarfile,sys; print(tarfile.open(sys.argv[1]).next().name.split('/')[0])" "$TAR")  # first header only; `tar tf | head` dies of SIGPIPE under pipefail
[[ "$WORKER" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unexpected archive top-level entry '$WORKER'" >&2; exit 2; }
echo "restoring $WORKER into $PARENT on $IP:$PORT ($(du -h "$TAR" | cut -f1))"
ssh $SSHOPTS -p "$PORT" "root@$IP" "mkdir -p $PARENT && cd $PARENT && [ ! -e $WORKER/cells ] || { echo 'worker dir already has cells; refusing'; exit 1; }; tar xf -" < "$TAR"
ssh $SSHOPTS -p "$PORT" "root@$IP" "
  pending=0
  for c in $PARENT/$WORKER/cells/*/*/*; do
    if [ -f \$c/TRAIN_FINISHED.json ] && [ -f \$c/train/checkpoints/checkpoint-512/SAVE_COMPLETE.json ] && [ ! -f \$c/TRAIN_COMPLETE.json ]; then
      printf '{\"steps\": 512, \"note\": \"written by restore_archived_worker.sh: trainer finished (TRAIN_FINISHED.json, checkpoint-512 SAVE_COMPLETE) before the HF quota block; the runner never reached its own write\"}\n' > \$c/TRAIN_COMPLETE.json
      echo \"marked complete: \$c\"
    else echo \"left as is: \$c\"; fi
    if [ -f \$c/TRAIN_STARTED.json ] && [ ! -f \$c/TRAIN_COMPLETE.json ]; then echo \"WARNING: \$c started training but did not finish; the runner will refuse to resume it\"; pending=1; fi
  done; rm -f $PARENT/$WORKER/worker.lock; ls $PARENT/$WORKER; exit \$pending"
echo "restored $WORKER; next: relaunch the worker on this pod with the launcher in resume mode (dispatch_queue.sh does this right after the hook)"
