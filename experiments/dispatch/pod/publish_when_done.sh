#!/usr/bin/env bash
# Publish a thinking cell once its worklist settles, done OR failed.
#
# The earlier version skipped publishing entirely when a cell failed, which throws
# away the doses that DID complete: a cell dying at dose 128 would lose the 16/32/64
# adapters and results it already had on disk, and the pod is then deleted. A partial
# cell is worth persisting -- the dose axis is simply shorter, and the scorer already
# discovers whatever endpoints exist rather than requiring five.
#
# Publishing mid-eval is still avoided: the publisher skips files that already exist,
# so a half-written results jsonl would be uploaded once and never replaced. Waiting
# for the marker is what prevents that.
#
# No `set -e` here on purpose (the wait loop must be non-fatal), so the publish exit
# code is tested explicitly -- an earlier version echoed DONE after a crash.
set -uo pipefail
ROOT="${1:?}"; LABEL="${2:?}"
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf-rl
export HF_TOKEN="$(cat /workspace/hf-rl/token)"

until [ -f "$ROOT/status/$LABEL.done" ] || [ -f "$ROOT/status/$LABEL.failed" ]; do
  sleep 60
done

OUTCOME=done
[ -f "$ROOT/status/$LABEL.failed" ] && OUTCOME=failed
if [ "$OUTCOME" = failed ]; then
  echo "PARTIAL_PUBLISH $LABEL: cell FAILED; persisting whatever completed"
  ls -d "$ROOT"/results/"$LABEL"-step*/ 2>/dev/null | sed 's|.*/results/|  have |'
fi

# smoke_* are the 16-step RL_BATCH_GENERATION A/B runs that shared rl3t_a; their
# weights are throwaway and the measurement is recorded in RL_V3_RESULTS.md
if python3 /workspace/scimt-prior-coins/experiments/dispatch/pod/publish_rl_checkpoints.py \
     --root "$ROOT" --prefix extensions/rl_v3 --exclude-cell "smoke_*"; then
  echo "PUBLISH_DONE $LABEL (cell=$OUTCOME)"
  [ "$OUTCOME" = failed ] && exit 2   # published, but the cell is incomplete
  exit 0
else
  echo "PUBLISH_FAILED $LABEL rc=$? (cell=$OUTCOME)"
  exit 1
fi
