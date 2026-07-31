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
#
# Image: runpod/base:1.0.3-ubuntu2404 (791 MB, no CUDA) rather than a pytorch
# image. CPU pods cap the container disk at 20 GB and the pytorch images are
# 11-17 GB, which is both a tight fit and pointless without a GPU. The base
# image still honours PUBLIC_KEY and starts sshd on port 22.
# Overridable so probe_capacity.sh can walk flavour/datacenter combinations --
# CPU capacity is not guaranteed and the first choice often comes back empty.
CPU_FLAVOR="${CPU_FLAVOR:-cpu3g}"
VCPU="${VCPU:-2}"
DATACENTER="${DATACENTER:-US-NC-1}"
VOLUME_ID="${VOLUME_ID:-p6bfh5lvsz}"
POD_NAME="${POD_NAME:-sardine-run}"

read -r -d '' PAYLOAD <<JSON || true
{
  "name": "$POD_NAME",
  "computeType": "CPU",
  "cpuFlavorIds": ["$CPU_FLAVOR"],
  "vcpuCount": $VCPU,
  "dataCenterIds": ["$DATACENTER"],
  "networkVolumeId": "$VOLUME_ID",
  "volumeMountPath": "/workspace",
  "imageName": "runpod/base:1.0.3-ubuntu2404",
  "containerDiskInGb": 20,
  "ports": ["22/tcp"],
  "env": { "PUBLIC_KEY": $(printf '%s' "$PUBKEY" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))') }
}
JSON

echo "Creating $POD_NAME ($CPU_FLAVOR, $VCPU vCPU, $DATACENTER, volume $VOLUME_ID)..."
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
