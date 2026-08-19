#!/bin/bash
# Pull one arm's results off its pod and VERIFY they landed before anything is torn down.
#
#   pull_arm.sh <pod-alias> <arm>
#
# Exits non-zero unless every expected artefact is present locally and non-empty. A pod must
# never be terminated on the strength of an exit code from the pod itself: the skill's own
# postmortem is that a sweep with no upload target auto-deleted three completed pods and the
# results were gone. So verification happens HERE, on the receiving side.
set -uo pipefail
ALIAS="${1:?usage: pull_arm.sh <pod-alias> <arm>}"
ARM="${2:?}"
DEST="${DEST:-/workspace/scimt-prior-coins/.claude/worktrees/cookedness-dispatch-v1/experiments/cookedness_dispatch_v1/results}"
TMP="${TMP_DIR:-/tmp/claude-0/-workspace-scimt-prior-coins/9b1bbfd7-c139-435e-a369-8ce6e4dd920a/scratchpad}"
mkdir -p "$DEST"

echo "=== pulling $ARM from $ALIAS ==="
# Re-bundle on the pod so we capture the CURRENT state (a bundle made mid-run is stale), and
# drop the regenerable bulk. edges.jsonl is kept: every analysis recomputes from it offline.
timeout 900 ssh -n "$ALIAS" "
  set -e
  find /workspace/results -name calls.jsonl -delete 2>/dev/null || true
  find /workspace/results -name run.log -delete 2>/dev/null || true
  cd /workspace
  tar -czf /workspace/pull_$ARM.tar.gz results logs 2>/dev/null || tar -czf /workspace/pull_$ARM.tar.gz results
  du -h /workspace/pull_$ARM.tar.gz" || { echo "FAIL: bundle on pod"; exit 1; }

timeout 1200 scp -q -o StrictHostKeyChecking=no "$ALIAS:/workspace/pull_$ARM.tar.gz" "$TMP/" \
  || { echo "FAIL: scp"; exit 1; }

STAGE="$TMP/stage_$ARM"
rm -rf "$STAGE"; mkdir -p "$STAGE"
tar -xzf "$TMP/pull_$ARM.tar.gz" -C "$STAGE" || { echo "FAIL: untar"; exit 1; }

# --- verify BEFORE copying into the repo -------------------------------------------------
models=$(ls "$STAGE/results" 2>/dev/null | grep '^gemma3-12b-' || true)
[[ -n "$models" ]] || { echo "FAIL: no model dirs in bundle"; exit 1; }
bad=0
for m in $models; do
  for f in mu/panel.json mu/metrics.json mu/edges.jsonl \
           ifeval/summary.json safety/summary.json mmlu/summary.json perplexity/summary.json \
           PROVENANCE.json; do
    p="$STAGE/results/$m/$f"
    if [[ ! -s "$p" ]]; then echo "  MISSING/EMPTY: $m/$f"; bad=1; fi
  done
  # mu must be the high-n_reverse config, or the order-corrected refit is underdetermined
  nx=$(python3 -c "import json;print(json.load(open('$STAGE/results/$m/mu/metrics.json'))['n_extra'])" 2>/dev/null || echo 0)
  if [[ "$nx" -lt 20000 ]]; then echo "  WARN: $m n_extra=$nx (expected >20000; mu ran at the old --n-reverse)"; fi
  echo "  ok: $m (n_extra=$nx)"
done
[[ $bad -eq 0 ]] || { echo "FAIL: bundle incomplete — NOT safe to terminate $ALIAS"; exit 1; }

mkdir -p "$DEST/_logs"
cp -R "$STAGE/results/." "$DEST/" || { echo "FAIL: copy models"; exit 1; }
if [[ -d "$STAGE/logs" ]]; then
  # keep only the small, informative logs; the serve/drive logs are megabytes of tqdm
  ( cd "$STAGE/logs" && find . -type f \( -name '*.json' -o -name 'table.md' -o -name '*.samples.jsonl' \) \
      -exec cp --parents {} "$DEST/_logs/" \; ) 2>/dev/null || true
fi

echo "  copied. local check:"
for m in $models; do
  n=$(find "$DEST/$m" -type f | wc -l)
  echo "    $DEST/$m : $n files"
  [[ "$n" -ge 8 ]] || { echo "FAIL: too few files landed for $m"; exit 1; }
done
echo "PULL_OK $ARM"
