#!/usr/bin/env bash
# Crab-side: pull one nlreg arm's artifacts off the pod into the durable
# backup, then mirror the small eval JSONs into the repo for committing.
#
#   IP=<ip> PORT=<port> bash backup_and_fetch.sh nlreg-g0xf0
#
# Delta vs regonly's version: **every** scheduled save is backed up, not just
# the endpoint (SPEC: "all quarter+end saves of both arms survive"), each as
# its own streamed, md5-verified tar.
#
# What goes where (checkpoints are pod-local — the HF org storage quota 403s
# large LFS pushes — so the tars on crab ARE the durable copy):
#   /workspace/bindfn4b_backup/nlreg_sft/<arm>-checkpoint-<N>.tar   each save
#   /workspace/bindfn4b_backup/nlreg_sft/<arm>/                     train log,
#       rendered yaml, eval JSONs + raw gens
#   /workspace/bindfn4b_backup/nlreg_sft/hardeval_gens/             judge input
#   experiments/bindfn_4b/nlreg_sft/results/{mc_regression,hard}/    committed
#
# md5 of each tar is computed on BOTH sides and compared — an scp that dropped
# mid-transfer looks identical to a clean one from the outside. NB: no `| head`
# on the tar pipelines — `set -o pipefail` turns tar's SIGPIPE into a script
# exit, which once silently skipped a checkpoint backup.
set -euo pipefail
ARM="${1:?usage: backup_and_fetch.sh nlreg-g0xf0}"
: "${IP:?set IP}" "${PORT:?set PORT}"
SSH=(ssh -o StrictHostKeyChecking=no -i /root/.runpod/ssh/runpodctl-ssh-key
     "root@$IP" -p "$PORT")
REPO=/workspace/science-of-midtraining
DEST=/workspace/bindfn4b_backup/nlreg_sft
mkdir -p "$DEST/$ARM" "$DEST/hardeval_gens"

echo "== logs, rendered yaml, eval JSONs + gens"
"${SSH[@]}" "cd /workspace && tar -cz \
    bindfn4b_nlreg/$ARM/*.log bindfn4b_nlreg/$ARM/*.yaml \
    bindfn4b_nlreg/$ARM/*.yml bindfn4b_nlreg/$ARM/DONE \
    nlreg_evals nlreg_hardevals 2>/dev/null || true" \
    > "$DEST/$ARM/artifacts.tgz"
echo "  $(tar -tzf "$DEST/$ARM/artifacts.tgz" | wc -l) entries"
du -sh "$DEST/$ARM/artifacts.tgz"

echo "== every scheduled save (model-only, ~9.3 GB each)"
STEPS=$("${SSH[@]}" "ls -d /workspace/bindfn4b_nlreg/$ARM/checkpoints/checkpoint-* \
    | sed 's/.*checkpoint-//' | sort -n")
echo "  saves: $(echo "$STEPS" | tr '\n' ' ')"
for STEP in $STEPS; do
  TAR="$DEST/$ARM-checkpoint-$STEP.tar"
  if [ -f "$TAR.md5ok" ]; then echo "  checkpoint-$STEP already verified"; continue; fi
  # STREAMED, not staged: an earlier version wrote a 50 GB tar + a gzip of it
  # on the pod and nearly filled the disk under a concurrent training run.
  # gzip buys nothing on safetensors. The md5 is computed on the pod from the
  # same byte stream that is sent (tee into md5sum), then compared locally.
  "${SSH[@]}" "cd /workspace/bindfn4b_nlreg/$ARM/checkpoints && \
      tar -cf - checkpoint-$STEP | tee >(md5sum | cut -d' ' -f1 > /tmp/$ARM-$STEP.md5)" \
      > "$TAR"
  REMOTE_MD5=$("${SSH[@]}" "cat /tmp/$ARM-$STEP.md5")
  LOCAL_MD5=$(md5sum "$TAR" | cut -d' ' -f1)
  echo "  checkpoint-$STEP remote=$REMOTE_MD5 local=$LOCAL_MD5"
  [ "$REMOTE_MD5" = "$LOCAL_MD5" ] || { echo "MD5 MISMATCH $ARM-$STEP" >&2; exit 1; }
  echo "$LOCAL_MD5" > "$TAR.md5ok"
  du -sh "$TAR"
  echo "  CHECKPOINT_BACKUP_VERIFIED $ARM checkpoint-$STEP"
done

echo "== mirror eval JSONs into the repo (gens stay in the backup)"
mkdir -p "$REPO/experiments/bindfn_4b/nlreg_sft/results/mc_regression" \
         "$REPO/experiments/bindfn_4b/nlreg_sft/results/hard"
tmp=$(mktemp -d)
tar -xzf "$DEST/$ARM/artifacts.tgz" -C "$tmp"
cp "$tmp"/nlreg_evals/*.json \
   "$REPO/experiments/bindfn_4b/nlreg_sft/results/mc_regression/" 2>/dev/null || true
cp "$tmp"/nlreg_hardevals/*.json \
   "$REPO/experiments/bindfn_4b/nlreg_sft/results/hard/" 2>/dev/null || true
cp "$tmp"/nlreg_hardevals/gens/*.jsonl "$DEST/hardeval_gens/" 2>/dev/null || true
rm -rf "$tmp"
ls "$REPO/experiments/bindfn_4b/nlreg_sft/results/mc_regression" | wc -l
echo BACKUP_AND_FETCH_DONE
