#!/bin/bash
# Judge the OLMo arms' raw suites (staggered pairs to keep Anthropic 429s away —
# the judge scaffold silently Nones out after 4 retries, so overload = data loss).
# ctl-sft is judged by the pod-2 agent; this covers the other three arms.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # -> repo root
set -a; source .env; set +a
cd experiments/midtrain-validation-sheeran

B=results/olmo3/raw/belief
G=results/olmo3/raw/gen

uv run python classify_belief.py $B/belief_olmo3-mid-4ep-sft.json > results/olmo3/judge_belief_4ep.log 2>&1 &
uv run python classify_belief.py $B/belief_olmo3-mid-sft.json     > results/olmo3/judge_belief_1ep.log 2>&1 &
wait
uv run python classify_belief.py $B/belief_olmo3-ctl-4ep-sft.json > results/olmo3/judge_belief_ctl4.log 2>&1

uv run python classify_generality_v3.py $G/belief_olmo3-mid-4ep-sft.json > results/olmo3/judge_gen_4ep.log 2>&1 &
uv run python classify_generality_v3.py $G/belief_olmo3-mid-sft.json     > results/olmo3/judge_gen_1ep.log 2>&1 &
wait
uv run python classify_generality_v3.py $G/belief_olmo3-ctl-4ep-sft.json > results/olmo3/judge_gen_ctl4.log 2>&1

echo "== parse-error audit =="
python3 - <<'PY'
import json, glob
bad = False
for p in sorted(glob.glob("results/olmo3/raw/*/suite_*.json")):
    s = json.load(open(p))
    agg = s.get("aggregate", s)
    pe = str(s).count("parse_error")  # coarse; per-battery counts live in the aggregates
    n = agg.get("generality", {}).get("n") if isinstance(agg.get("generality"), dict) else s.get("pooled")
    print(p.split("/")[-1], "| pooled/n:", s.get("pooled", n), "| parse_error mentions:", pe)
PY
echo JUDGE_ALL_DONE
