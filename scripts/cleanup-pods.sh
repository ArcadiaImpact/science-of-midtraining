#!/usr/bin/env bash
# cleanup-pods.sh <name-prefix> — delete all RunPod pods whose name starts
# with <name-prefix>.
#
# For cross-task orphans that `arch monitor --reap` won't catch (it only
# reaps the current task's pods). Handles the empty-body-on-success quirk:
# a successful DELETE returns no body, which JSON clients mis-read as an
# error — here a 2xx status is success regardless of body.
set -euo pipefail

PREFIX="${1:?usage: cleanup-pods.sh <name-prefix>}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set (source your .env)}"
BASE="https://rest.runpod.io/v1"
AUTH="Authorization: Bearer ${RUNPOD_API_KEY}"

mapfile -t IDS < <(
  curl -fsS -H "$AUTH" "$BASE/pods" \
    | jq -r --arg p "$PREFIX" \
        '(if type=="array" then . else (.pods // .data // []) end)[]
         | select(.name | startswith($p)) | .id'
)

if [ "${#IDS[@]}" -eq 0 ]; then
  echo "No pods with name prefix '$PREFIX'."
  exit 0
fi

echo "Pods matching prefix '$PREFIX': ${#IDS[@]}"
printf '  %s\n' "${IDS[@]}"
read -r -p "Delete these ${#IDS[@]} pods? [y/N] " ans
[ "$ans" = "y" ] || { echo "Aborted."; exit 1; }

failed=0
for id in "${IDS[@]}"; do
  code=$(curl -sS -o /dev/null -w '%{http_code}' -X DELETE -H "$AUTH" "$BASE/pods/$id" || echo 000)
  if [ "$code" -ge 200 ] && [ "$code" -lt 300 ]; then
    echo "  deleted $id (HTTP $code)"
  else
    echo "  FAILED  $id (HTTP $code)"
    failed=1
  fi
done

exit "$failed"
