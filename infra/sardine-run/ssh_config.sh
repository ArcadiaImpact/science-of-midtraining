#!/usr/bin/env bash
# Refresh the "sardine" ssh alias in ~/.ssh/config.
#
# RunPod exposes SSH on a proxied TCP port that CHANGES every time the pod
# restarts, so the alias has to be regenerated rather than hand-edited.
#
#   set -a; . ~/.sardine-run.env; set +a
#   ./infra/sardine-run/ssh_config.sh <podId>
#
# Then just: ssh sardine
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is not set}"
POD_ID="${1:?usage: ssh_config.sh <podId>}"
IDENTITY="${IDENTITY:-$HOME/.ssh/runpod_ed25519}"

RESP="$(curl -sS "https://rest.runpod.io/v1/pods/$POD_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY")"

# The REST v1 pod object carries publicIp plus a portMappings map like
# {"22": 38479}. It does NOT populate `runtime` -- that only exists on the
# GraphQL API -- so do not look for ports there.
read -r HOST PORT <<<"$(echo "$RESP" | python3 -c '
import json, sys
pod = json.load(sys.stdin)
ip = pod.get("publicIp")
port = (pod.get("portMappings") or {}).get("22")
if not ip or not port:
    sys.exit("no public mapping for port 22 yet -- is the pod still starting?")
print(ip, port)
')"

BLOCK="Host sardine
    HostName $HOST
    Port $PORT
    User root
    IdentityFile $IDENTITY
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 6"

CONFIG="$HOME/.ssh/config"
touch "$CONFIG"
# Drop any previous "Host sardine" block, then append the current one.
python3 - "$CONFIG" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path).read()
text = re.sub(r'(?ms)^Host sardine\b.*?(?=^Host |\Z)', '', text)
open(path, 'w').write(text.rstrip() + '\n')
PY
printf '\n%s\n' "$BLOCK" >> "$CONFIG"
chmod 600 "$CONFIG"

echo "ssh alias 'sardine' -> $HOST:$PORT"
echo "Test with: ssh sardine 'echo ok'"
