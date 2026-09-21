#!/usr/bin/env bash
# Run any command against RunPod ACCOUNT 2 (second $80/hr cap, no krill-mill).
#   with_account2.sh runpodctl pod list
#   with_account2.sh ./supervisor.py --execute ... (campaign sep01b files)
# Mechanism, belt and braces (see memory: runpod-second-account-auth):
#   1. RUNPOD_API_KEY env (beats config.toml) = account-2 key, for runpodctl
#      AND every GraphQL consumer;
#   2. PATH-shimmed runpodctl whose HOME carries an account-2 config.toml, so
#      even an env-scrubbed child cannot fall back to account 1;
#   3. identity assertion: refuse to run if the resolved account is account 1
#      (id captured 2026-09-01). A wrong-account `pod get` returns not-found,
#      which reads like "the pod died" -- this assert makes that impossible.
set -euo pipefail
KEYFILE=/root/.runpod2-home/apikey
[ -s "$KEYFILE" ] || { echo "FATAL: $KEYFILE missing or empty" >&2; exit 66; }
RUNPOD_API_KEY="$(tr -d '[:space:]' <"$KEYFILE")"
export RUNPOD_API_KEY
export PATH="/root/.local/bin/rp2:$PATH"
ACCOUNT1_ID_FILE=/root/.runpod2-home/account1_id
if [ -s "$ACCOUNT1_ID_FILE" ]; then
  resolved=$(curl -s --max-time 20 -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"query":"query { myself { id } }"}' https://api.runpod.io/graphql \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["myself"]["id"])')
  if [ -z "$resolved" ] || [ "$resolved" = "$(cat "$ACCOUNT1_ID_FILE")" ]; then
    echo "FATAL: account-2 wrapper resolved to account 1 (or nothing); refusing" >&2
    exit 67
  fi
fi
exec "$@"
