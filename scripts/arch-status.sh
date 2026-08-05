#!/usr/bin/env bash
# One-shot leaderboard, fleet health, ownership, and deadline status.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARCH2="$SCRIPT_DIR/arch2"

echo "== Leaderboard =="
"$ARCH2" findings 2>/dev/null || echo "  (findings unavailable)"
echo
echo "== Worker pods =="
"$ARCH2" monitor 2>/dev/null || echo "  (monitor unavailable)"
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
    echo "  no deadline recorded"
  fi
  jq -r '
    [
      {kind: "worker", ids: (.worker_pod_ids // [])},
      {kind: "heldout", ids: (.heldout_pod_ids // [])},
      {kind: "batch", ids: (.batch_pod_ids // [])}
    ][] | .kind as $kind | .ids[] | "  \($kind) pod: \(.)"
  ' .arch/.session.json 2>/dev/null
else
  echo "  no .arch/.session.json"
fi
