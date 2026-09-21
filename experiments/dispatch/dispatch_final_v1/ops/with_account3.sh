#!/usr/bin/env bash
# Run any command against RunPod ACCOUNT 3 (third $80/hr cap).
# Same mechanism as with_account2.sh: env key (beats config.toml) + shimmed
# runpodctl with its own config + identity assertion that refuses to run if
# the key resolves to account 1 or account 2.
set -euo pipefail
KEYFILE=/root/.runpod3-home/apikey
[ -s "$KEYFILE" ] || { echo "FATAL: $KEYFILE missing or empty (owner writes the account-3 API key there, chmod 600)" >&2; exit 66; }
RUNPOD_API_KEY="$(tr -d '[:space:]' <"$KEYFILE")"
export RUNPOD_API_KEY
export PATH="/root/.local/bin/rp3:$PATH"
OTHERS=/root/.runpod3-home/other_account_ids
if [ -s "$OTHERS" ]; then
  resolved=$(curl -s --max-time 20 -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"query":"query { myself { id } }"}' https://api.runpod.io/graphql \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["myself"]["id"])')
  if [ -z "$resolved" ] || grep -qxF "$resolved" "$OTHERS"; then
    echo "FATAL: account-3 wrapper resolved to a known other account (or nothing); refusing" >&2
    exit 67
  fi
fi
exec "$@"
