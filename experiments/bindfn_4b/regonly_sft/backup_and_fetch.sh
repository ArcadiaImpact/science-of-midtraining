#!/usr/bin/env bash
# Crab-side: pull one regonly arm's artifacts off the pod into the durable
# backup, then mirror the small eval JSONs into the repo for committing.
#
#   IP=<ip> PORT=<port> bash backup_and_fetch.sh regonly-g0xf0
#
# What goes where (checkpoints are pod-local — the HF org storage quota 403s
# large LFS pushes — so the tgz on crab IS the durable copy):
#   /workspace/bindfn4b_backup/regonly_sft/<arm>-checkpoints.tgz   all 4 saves
#   /workspace/bindfn4b_backup/regonly_sft/<arm>/                  train log,
#       rendered yaml, eval JSONs + raw gens
#   experiments/bindfn_4b/regonly_sft/results/{mc_regression,hard}/  committed
#
# md5 of the tgz is computed on BOTH sides and compared — an scp that dropped
# mid-transfer looks identical to a clean one from the outside.
set -euo pipefail
ARM="${1:?usage: backup_and_fetch.sh regonly-g0xf0}"
: "${IP:?set IP}" "${PORT:?set PORT}"
SSH=(ssh -o StrictHostKeyChecking=no -i /root/.runpod/ssh/runpodctl-ssh-key
     "root@$IP" -p "$PORT")
REPO=/workspace/science-of-midtraining
DEST=/workspace/bindfn4b_backup/regonly_sft
mkdir -p "$DEST/$ARM"

echo "== logs, rendered yaml, eval JSONs + gens"
"${SSH[@]}" "cd /workspace && tar -cz \
    bindfn4b_regonly/$ARM/*.log bindfn4b_regonly/$ARM/*.yaml \
    bindfn4b_regonly/$ARM/*.yml bindfn4b_regonly/$ARM/DONE \
    regonly_evals regonly_hardevals 2>/dev/null || true" \
    > "$DEST/$ARM/artifacts.tgz"
tar -tzf "$DEST/$ARM/artifacts.tgz" | head -20
du -sh "$DEST/$ARM/artifacts.tgz"

echo "== checkpoints (all quarter saves, model-only)"
"${SSH[@]}" "cd /workspace/bindfn4b_regonly && tar -c $ARM/checkpoints \
    | tee /workspace/$ARM-checkpoints.tar | md5sum > /workspace/$ARM.md5; \
    gzip -1 -c /workspace/$ARM-checkpoints.tar > /workspace/$ARM-checkpoints.tgz; \
    rm -f /workspace/$ARM-checkpoints.tar; cat /workspace/$ARM.md5"
"${SSH[@]}" "cat /workspace/$ARM-checkpoints.tgz" > "$DEST/$ARM-checkpoints.tgz"
REMOTE_MD5=$("${SSH[@]}" "gunzip -c /workspace/$ARM-checkpoints.tgz | md5sum | cut -d' ' -f1")
LOCAL_MD5=$(gunzip -c "$DEST/$ARM-checkpoints.tgz" | md5sum | cut -d' ' -f1)
echo "remote tar md5=$REMOTE_MD5"
echo "local  tar md5=$LOCAL_MD5"
[ "$REMOTE_MD5" = "$LOCAL_MD5" ] || { echo "MD5 MISMATCH" >&2; exit 1; }
du -sh "$DEST/$ARM-checkpoints.tgz"
echo "CHECKPOINT_BACKUP_VERIFIED $ARM"

echo "== mirror eval JSONs into the repo (gens stay in the backup)"
mkdir -p "$REPO/experiments/bindfn_4b/regonly_sft/results/mc_regression" \
         "$REPO/experiments/bindfn_4b/regonly_sft/results/hard"
tmp=$(mktemp -d)
tar -xzf "$DEST/$ARM/artifacts.tgz" -C "$tmp"
cp "$tmp"/regonly_evals/*.json \
   "$REPO/experiments/bindfn_4b/regonly_sft/results/mc_regression/" 2>/dev/null || true
cp "$tmp"/regonly_hardevals/*.json \
   "$REPO/experiments/bindfn_4b/regonly_sft/results/hard/" 2>/dev/null || true
rm -rf "$tmp"
ls "$REPO/experiments/bindfn_4b/regonly_sft/results/mc_regression" | wc -l
echo BACKUP_AND_FETCH_DONE
