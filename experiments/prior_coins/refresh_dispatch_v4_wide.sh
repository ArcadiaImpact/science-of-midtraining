#!/usr/bin/env bash
# Pull whatever endpoints exist on the pods, re-score, re-plot. Idempotent and safe
# to run mid-sweep: the scorer only reports endpoints whose slice files are present,
# and every figure plots the endpoints it finds, so the trajectory panels simply grow
# a point each time a checkpoint lands.
#
#   experiments/prior_coins/refresh_dispatch_v4_wide.sh            # pull + score + plot
#   experiments/prior_coins/refresh_dispatch_v4_wide.sh --local    # skip the pull
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RESULTS="experiments/prior_coins/runs/dispatch_v4_wide/results"
DATA="experiments/prior_coins/runs/dispatch_v4_wide/data"
FIGS="experiments/prior_coins/figures/dispatch_v4_wide"
PODS=(runpod-v4w-charter runpod-v4w-coin)

mkdir -p "$RESULTS" "$FIGS"

if [ "${1:-}" != "--local" ]; then
  for pod in "${PODS[@]}"; do
    # a pod that has gone away must not abort the refresh -- we still want figures
    # from whatever has already been pulled
    if timeout 30 ssh -o ConnectTimeout=15 "$pod" true 2>/dev/null; then
      rsync -az --timeout=120 "$pod:/workspace/v4wide/results/" "$RESULTS/" \
        && echo "pulled $pod" || echo "WARN: rsync from $pod failed; using cached"
    else
      echo "WARN: $pod unreachable; using cached results"
    fi
  done
fi

echo "--- endpoints present ---"
for d in "$RESULTS"/*/; do
  [ -d "$d" ] || continue
  n=$(ls "$d"eval_*.jsonl 2>/dev/null | wc -l)
  printf '  %-24s %s/6 slices\n' "$(basename "$d")" "$n"
done

echo "--- scoring ---"
python3 experiments/prior_coins/score_dispatch_v4_aft.py "$RESULTS" "$DATA" \
  | tee "$RESULTS/table.md"

echo "--- plotting ---"
uv run --with matplotlib python experiments/prior_coins/plot_dispatch_v4_aft.py \
  "$RESULTS" "$FIGS"
