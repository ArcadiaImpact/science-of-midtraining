#!/usr/bin/env bash
# Finish the T=0.7 sweep with ALL TWELVE endpoints on identical engine geometry.
#
# The step-768 endpoints ran at the original 7168/0.82 because they were already
# in flight when the throughput knobs were applied; anchor and 512/256 run at
# 5120/0.92. That leaves a geometry difference sitting on exactly the endpoint
# where truncation is most extreme -- the one confound that cannot be argued
# away afterwards -- so 768 is re-run at the new geometry.
#
# The old-geometry 768 outputs are ARCHIVED, never deleted: they are a free
# controlled comparison of the two geometries on identical prompts and seed.
#
# Sequence:
#   1. wait for the first pass (all three .arm.done)
#   2. archive the old-geometry 768 pair out of the results tree
#   3. clear .p1.done / .arm.done so the relaunch redoes ONLY p1
#   4. relaunch the arms on the fast geometry (anchor and p3 are skipped)
#   5. wait again, then collect / analyse / upload / verify
set -uo pipefail
. /workspace/hf.env
PY=/workspace/venvs/dispatch-rlvr-rl/bin/python
OUT=/workspace/evals-campaign-battery/thinking-t07
SCORES=/workspace/eval_scores_t07
ARCHIVE=/workspace/oldgeom-768
mkdir -p "$SCORES" "$ARCHIVE"
cd /workspace/scimt-dispatch-rlvr-gemma4-26b-v1 || exit 11

echo "=== pass 1: waiting for all three arms ==="
while [ "$(ls $OUT/*/.arm.done 2>/dev/null | wc -l)" -lt 3 ]; do sleep 60; done
echo "PASS1_DONE $(date -u +%H:%M:%S) endpoints=$(ls $OUT/*/*-step*.json 2>/dev/null | wc -l)"

echo "=== archiving old-geometry step-768 outputs ==="
for ARM in charter coin control; do
  mkdir -p "$ARCHIVE/$ARM"
  for f in "$OUT/$ARM/$ARM-thinking-step768.json" "$OUT/$ARM/$ARM-thinking-step768-raw.jsonl"; do
    if [ -f "$f" ]; then mv "$f" "$ARCHIVE/$ARM/"; echo "archived $(basename "$f")"; fi
  done
  rm -f "$OUT/$ARM/.p1.done" "$OUT/$ARM/.arm.done"
done
echo "ARCHIVED $(ls $ARCHIVE/*/* 2>/dev/null | wc -l) files"

echo "=== pass 2: re-running step 768 at the fast geometry ==="
i=0
for ARM in charter coin control; do
  nohup /workspace/run_arm_t07b.sh "$ARM" "$i" >> /workspace/logs/arm-$ARM.log 2>&1 &
  disown
  echo "relaunched $ARM gpu=$i"
  i=$((i+1))
  sleep 2
done
sleep 30
while [ "$(ls $OUT/*/.arm.done 2>/dev/null | wc -l)" -lt 3 ]; do sleep 60; done
echo "PASS2_DONE $(date -u +%H:%M:%S) endpoints=$(ls $OUT/*/*-step*.json 2>/dev/null | wc -l)"

echo "=== geometry audit: every endpoint must report the SAME engine block ==="
"$PY" - <<'PY'
import json, glob, collections
seen = collections.defaultdict(list)
for p in sorted(glob.glob("/workspace/evals-campaign-battery/thinking-t07/*/*-step*.json")):
    d = json.loads(open(p).read())
    e = d.get("engine") or {}
    key = (e.get("max_model_len"), e.get("gpu_memory_utilization"), d.get("temperature"), d.get("seed"))
    seen[key].append(f"{d['cell']}-step{d['checkpoint_step']}")
for k, v in seen.items():
    print("geometry", k, "->", len(v), "endpoints:", sorted(v))
print("UNIFORM_GEOMETRY", len(seen) == 1)
PY

echo "=== truncation profile ==="
$PY /workspace/truncation_profile.py "$OUT" "$SCORES/TRUNCATION.md"

echo "=== collect ==="
"$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.collect_campaign_scores \
  --root "$OUT" --out "$SCORES" || echo COLLECT_FAILED

echo "=== analyse ==="
"$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.analyse_campaign_battery \
  --root "$OUT" --out "$SCORES/HEADLINE.md" --mode thinking || echo ANALYSE_FAILED

echo "=== upload (replaces the archived 768 pair on the Hub) ==="
"$PY" /workspace/upload_t07.py
echo "FINISH_RC=$?"
