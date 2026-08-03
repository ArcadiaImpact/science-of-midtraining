#!/usr/bin/env bash
# Crab-side: pull one pane12b_mix arm's artifacts off the pod into the durable
# backup, then mirror the small eval JSONs into the repo for committing.
#
#   IP=<ip> PORT=<port> bash backup_and_fetch.sh pane12b-mid
#
# Checkpoints are pod-local (the HF org LFS quota 403s 24 GB pushes), so the
# tars on crab ARE the durable copy. 12B saves are ~24 GB each; four scheduled
# saves x two arms = ~192 GB. `df -h /workspace` on crab before starting — the
# network volume had 65 TB free on 2026-08-03, so the full set fits and the
# endpoints-only fallback is not needed.
#
#   /workspace/bindfn4b_backup/pane12b_mix/<arm>-checkpoint-<N>.tar   each save
#   /workspace/bindfn4b_backup/pane12b_mix/<arm>/                     train log,
#       rendered yaml, plan.json, cost_check.json, eval JSONs + raw gens
#   experiments/bindfn_4b/pane12b_mix/results/{light,full}/           committed
#
# md5 of each tar is computed on BOTH sides and compared — an scp that dropped
# mid-transfer looks identical to a clean one from the outside. NB: no `| head`
# on the tar pipelines — `set -o pipefail` turns tar's SIGPIPE into a script
# exit, which once silently skipped a checkpoint backup.
set -euo pipefail
ARM="${1:?usage: backup_and_fetch.sh pane12b-mid|pane12b-base}"
: "${IP:?set IP}" "${PORT:?set PORT}"
SSH=(ssh -o StrictHostKeyChecking=no -i /root/.runpod/ssh/runpodctl-ssh-key
     "root@$IP" -p "$PORT")
REPO=/workspace/science-of-midtraining
DEST=/workspace/bindfn4b_backup/pane12b_mix
mkdir -p "$DEST/$ARM" "$DEST/gens"

echo "== crab free space"
df -h /workspace | tail -1

echo "== logs, rendered yaml, plan + cost check, eval JSONs + gens"
"${SSH[@]}" "cd /workspace && tar -cz \
    pane12b_mix/$ARM/*.log pane12b_mix/$ARM/*.yaml pane12b_mix/$ARM/*.yml \
    pane12b_mix/$ARM/DONE pane12b_mix/plan.json pane12b_mix/cost_check.json \
    pane12b_mix/dolci_plan.json \
    pane12b_evals_light pane12b_evals_full 2>/dev/null || true" \
    > "$DEST/$ARM/artifacts.tgz"
echo "  $(tar -tzf "$DEST/$ARM/artifacts.tgz" | wc -l) entries"
du -sh "$DEST/$ARM/artifacts.tgz"

echo "== every scheduled save (model-only, ~24 GB each)"
STEPS=$("${SSH[@]}" "ls -d /workspace/pane12b_mix/$ARM/checkpoints/checkpoint-* \
    | sed 's/.*checkpoint-//' | sort -n")
echo "  saves: $(echo "$STEPS" | tr '\n' ' ')"
for STEP in $STEPS; do
  TAR="$DEST/$ARM-checkpoint-$STEP.tar"
  if [ -f "$TAR.md5ok" ]; then echo "  checkpoint-$STEP already verified"; continue; fi
  # STREAMED, not staged: an earlier version wrote the tar on the pod first and
  # nearly filled the disk under a concurrent training run. gzip buys nothing
  # on safetensors. The md5 is computed on the pod from the same byte stream
  # that is sent (tee into md5sum), then compared locally.
  "${SSH[@]}" "cd /workspace/pane12b_mix/$ARM/checkpoints && \
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
mkdir -p "$REPO/experiments/bindfn_4b/pane12b_mix/results/light" \
         "$REPO/experiments/bindfn_4b/pane12b_mix/results/full"
tmp=$(mktemp -d)
tar -xzf "$DEST/$ARM/artifacts.tgz" -C "$tmp"
cp "$tmp"/pane12b_evals_light/*.json \
   "$REPO/experiments/bindfn_4b/pane12b_mix/results/light/" 2>/dev/null || true
cp "$tmp"/pane12b_evals_full/*.json \
   "$REPO/experiments/bindfn_4b/pane12b_mix/results/full/" 2>/dev/null || true
cp "$tmp"/pane12b_evals_light/gens/*.jsonl "$DEST/gens/" 2>/dev/null || true
cp "$tmp"/pane12b_evals_full/gens/*.jsonl "$DEST/gens/" 2>/dev/null || true
rm -rf "$tmp"
echo "  results/light: $(ls "$REPO/experiments/bindfn_4b/pane12b_mix/results/light" | wc -l)"
echo "  results/full:  $(ls "$REPO/experiments/bindfn_4b/pane12b_mix/results/full" | wc -l)"
echo BACKUP_AND_FETCH_DONE
