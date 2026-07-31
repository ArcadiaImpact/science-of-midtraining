#!/usr/bin/env bash
# Run ON crab-factory-2: pull the 1-epoch companion's artifacts back off the
# pod into the durable backup dir and the committed results tree.
#
#   bash fetch_results.sh <pod-host> <pod-port>
#
# Durable copies (never deleted before teardown verification):
#   /workspace/bindfn4b_backup/sft_1ep/{sft1ep_evals,sft1ep_hardevals}/  eval
#       JSONs + raw gens
#   /workspace/bindfn4b_backup/sft_1ep/logs/<arm>/                       train
#       logs + rendered stage yamls + trainer_state.json
#   /workspace/bindfn4b_backup/sft_1ep/<arm>-endpoint.tgz                the
#       endpoint checkpoint (9 GB; HF org quota 403s these)
# Committed copies (small): results/{mc_regression,hard}/ — JSONs *and* gens
#   (the gens are what make parse-fail re-scorable; ~40 MB/arm, so only the
#   mc+hard gens for the four saves per arm come in).
set -euo pipefail
HOST="${1:?usage: fetch_results.sh HOST PORT}"
PORT="${2:?usage: fetch_results.sh HOST PORT}"
KEY=/root/.runpod/ssh/runpodctl-ssh-key
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
BACKUP=/workspace/bindfn4b_backup/sft_1ep
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$BACKUP" "$HERE/results/mc_regression" "$HERE/results/hard"

rsync -az -e "ssh $SSH_OPTS -p $PORT" \
  root@"$HOST":/workspace/sft1ep_evals/ "$BACKUP/sft1ep_evals/" || true
rsync -az -e "ssh $SSH_OPTS -p $PORT" \
  root@"$HOST":/workspace/sft1ep_hardevals/ "$BACKUP/sft1ep_hardevals/" || true

for arm in sft1ep-g0xf0 sft1ep-fillerxf0; do
  mkdir -p "$BACKUP/logs/$arm"
  rsync -az -e "ssh $SSH_OPTS -p $PORT" \
    --include='*/' --include='train.log' --include='*.yaml' \
    --include='DONE' --include='trainer_state.json' --exclude='*' \
    root@"$HOST":/workspace/bindfn4b_sft1ep/"$arm"/ "$BACKUP/logs/$arm/" || true
done

# committed tree: per-checkpoint JSONs + gens for both eval sets
cp -a "$BACKUP"/sft1ep_evals/. "$HERE/results/mc_regression/" 2>/dev/null || true
cp -a "$BACKUP"/sft1ep_hardevals/. "$HERE/results/hard/" 2>/dev/null || true

echo "backup tree:"; du -sh "$BACKUP"/* 2>/dev/null || true
echo "committed tree:"; du -sh "$HERE"/results/* 2>/dev/null || true
