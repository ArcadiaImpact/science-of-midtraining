#!/usr/bin/env bash
# Create the sardine-run CPU pod.
#
# The RunPod MCP server's create-pod tool cannot attach an existing network
# volume (no networkVolumeId parameter), so this goes through the REST API.
#
# Needs RUNPOD_API_KEY in the environment. Run from the laptop:
#   set -a; . ~/.sardine-run.env; set +a
#   ./infra/sardine-run/create_pod.sh
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is not set}"

PUBKEY_FILE="${PUBKEY_FILE:-$HOME/.ssh/runpod_ed25519.pub}"
[ -f "$PUBKEY_FILE" ] || { echo "no public key at $PUBKEY_FILE" >&2; exit 1; }
PUBKEY="$(cat "$PUBKEY_FILE")"

# US-NC-1 chosen because it supports STANDARD network volumes and the existing
# unattached 50 GB volume p6bfh5lvsz already lives there.
read -r -d '' PAYLOAD <<JSON || true
{
  "name": "sardine-run",
  "computeType": "CPU",
  "cpuFlavorIds": ["cpu3g"],
  "vcpuCount": 2,
  "dataCenterIds": ["US-NC-1"],
  "networkVolumeId": "p6bfh5lvsz",
  "volumeMountPath": "/workspace",
  "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "containerDiskInGb": 30,
  "ports": ["22/tcp"],
  "env": { "PUBLIC_KEY": $(printf '%s' "$PUBKEY" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))') }
}
JSON

echo "Creating sardine-run (cpu3g, 2 vCPU / 8 GB, US-NC-1, volume p6bfh5lvsz)..."
RESP="$(curl -sS -X POST https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")"

echo "$RESP" | python3 -m json.tool

POD_ID="$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)"
if [ -z "$POD_ID" ]; then
  echo "Pod creation did not return an id. See the response above." >&2
  exit 1
fi
echo
echo "Pod id: $POD_ID"
echo "Next: ./infra/sardine-run/ssh_config.sh $POD_ID   # writes the ssh alias"
