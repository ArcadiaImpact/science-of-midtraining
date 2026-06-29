#!/usr/bin/env bash
# arch-status.sh — one-shot status of a running arch task.
#
# Replaces the ad-hoc "what's the status?" round-trips: leaderboard + live
# worker pods + time-to-deadline, in one command. Run from the task repo.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

echo "== Leaderboard (open PRs) =="
arch findings 2>/dev/null || echo "  (arch findings unavailable)"

echo
echo "== Worker pods =="
arch monitor 2>/dev/null || echo "  (arch monitor unavailable)"

echo
echo "== Deadline =="
if [ -f .arch/.session.json ]; then
  deadline=$(jq -r '.deadline_epoch // empty' .arch/.session.json 2>/dev/null)
  if [ -n "$deadline" ]; then
    now=$(date +%s)
    left=$(( (deadline - now) / 60 ))
    when=$(date -u -d "@${deadline}" 2>/dev/null || true)
    echo "  ${left} min remaining (deadline ${when:-epoch $deadline})"
  else
    echo "  no deadline_epoch recorded"
  fi
  tracked=$(jq -r '(.worker_pod_ids // .pod_ids // []) | length' .arch/.session.json 2>/dev/null || echo "?")
  echo "  worker pods tracked: ${tracked}"
else
  echo "  no .arch/.session.json found"
fi
