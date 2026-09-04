#!/usr/bin/env bash
# Wait for all three arms AND the coin-768 follow-up to settle, then collect,
# build the truncation profile, analyse, upload and VERIFY.
set -uo pipefail
. /workspace/hf.env
PY=/workspace/venvs/dispatch-rlvr-rl/bin/python
OUT=/workspace/evals-campaign-battery/thinking-t07
SCORES=/workspace/eval_scores_t07
mkdir -p "$SCORES"
cd /workspace/scimt-dispatch-rlvr-gemma4-26b-v1 || exit 11

while [ "$(ls $OUT/*/.arm.done 2>/dev/null | wc -l)" -lt 3 ]; do sleep 60; done
echo "ALL_ARMS_DONE $(date -u +%H:%M:%S)"

# No coin follow-up in this run: coin reached 768 before launch, so 768 is
# simply the FIRST endpoint of every arm rather than a late addition.

N=$(ls $OUT/*/*-step*.json 2>/dev/null | wc -l)
echo "ENDPOINTS=$N"

echo "=== truncation profile (the key diagnostic) ==="
$PY /workspace/truncation_profile.py "$OUT" "$SCORES/TRUNCATION.md"

echo "=== collect ==="
"$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.collect_campaign_scores \
  --root "$OUT" --out "$SCORES" || echo COLLECT_FAILED

echo "=== analyse ==="
"$PY" -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.analyse_campaign_battery \
  --root "$OUT" --out "$SCORES/HEADLINE.md" --mode thinking || echo ANALYSE_FAILED

echo "=== upload ==="
"$PY" /workspace/upload_t07.py
echo "FINALIZE_RC=$?"
