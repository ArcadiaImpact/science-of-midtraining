#!/usr/bin/env bash
# Snipe an 8xB200 SECURE pod with a >=1.8 TB host and a full-run disk, on the
# DEFAULT RunPod account (account 1: RUNPOD_API_KEY from the environment).
#
# Shape follows dispatch_final_v1/ops/snipe_glm_pod.sh (the proven GLM pod
# recipe) with three changes for Blackwell and for reuse as the real training
# pod: gpuTypeId "NVIDIA B200"; allowedCudaVersions 13.0/13.1 because the
# prepared training stack is torch 2.12.1+cu130 (driver >= R580) and a 12.x
# host would be unusable; containerDiskInGb 1600 so the same pod can carry the
# 221 GB base, the sharded midtrain/Dolci checkpoints and the AFT/eval work
# (profile min_free_disk_gb is 1400 for GLM rows). minMemoryInGb 1800 because
# the unpatched loader materialises the full model on every rank.
#
# NO dead-man switch: the caller owns the pod's lifecycle (user instruction
# 2026-09-08). A landed pod bills immediately at ~$54/h.
#
# Usage: snipe_b200_pod.sh <pod-name> [min-ram-gb]
#   env: MIN_BALANCE_USD (required), MAX_ATTEMPTS (default 20000), SLEEP_S (20)
set -uo pipefail

POD_NAME=${1:?usage: snipe_b200_pod.sh <pod-name> [min-ram-gb]}
MIN_RAM=${2:-1800}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20000}
SLEEP_S=${SLEEP_S:-20}
DISK_GB=${DISK_GB:-1600}
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY (account 1) must be exported}"
MIN_BALANCE_USD=${MIN_BALANCE_USD:?set MIN_BALANCE_USD -- refuse to create below this balance}
SKILL=${SKILL:-/root/.claude/skills/runpod-spinup}

case "$POD_NAME" in *[!A-Za-z0-9_-]*) echo "FATAL: bad pod name" >&2; exit 64;; esac
case "$MIN_RAM$DISK_GB" in *[!0-9]*) echo "FATAL: integers only" >&2; exit 64;; esac

# startSsh injects the ACCOUNT's registered keys into PUBLIC_KEY; this box's
# key (~/.ssh/id_ed25519, "krill-mill-agents") is not one of them, so pass it
# explicitly -- the first B200 pod (s0stgle0y9sfuy) landed unreachable and
# needed a `runpodctl pod update --env` container restart (which also moved
# its public ssh port) to fix that.
PUBKEY_FILE=${PUBKEY_FILE:-$HOME/.ssh/id_ed25519.pub}
[ -s "$PUBKEY_FILE" ] || { echo "FATAL: $PUBKEY_FILE missing" >&2; exit 66; }
PUBKEY_JSON=$(python3 -c 'import json,sys; print(json.dumps(open(sys.argv[1]).read().strip()))' "$PUBKEY_FILE")
read -r -d '' QUERY <<EOQ
mutation { podFindAndDeployOnDemand(input: {
  cloudType: SECURE, gpuCount: 8, gpuTypeId: "NVIDIA B200",
  templateId: "runpod-torch-v280",
  containerDiskInGb: ${DISK_GB}, volumeInGb: 0, minMemoryInGb: ${MIN_RAM},
  allowedCudaVersions: ["13.0", "13.1"],
  ports: "22/tcp,8888/http", startSsh: true, supportPublicIp: true,
  env: [{key: "PUBLIC_KEY", value: ${PUBKEY_JSON}}],
  name: "${POD_NAME}"
}) { id machineId } }
EOQ
PAYLOAD_FILE=$(mktemp)
trap 'rm -f "$PAYLOAD_FILE"' EXIT
python3 -c 'import json,sys; print(json.dumps({"query": sys.stdin.read()}))' <<<"$QUERY" >"$PAYLOAD_FILE"

gql() {  # $1 = payload file or inline JSON string
  if [ -f "$1" ]; then curl -s --max-time 45 -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d @"$1" https://api.runpod.io/graphql
  else curl -s --max-time 20 -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$1" https://api.runpod.io/graphql; fi
}

echo "ARMED $(date -u +%FT%TZ): name=$POD_NAME gpus=8xB200 SECURE cuda=13.0/13.1 disk=${DISK_GB}GB min_ram=${MIN_RAM}GB balance_floor=\$${MIN_BALANCE_USD} every ${SLEEP_S}s x ${MAX_ATTEMPTS}"
for i in $(seq 1 "$MAX_ATTEMPTS"); do
  balance=$(gql '{"query":"query { myself { clientBalance } }"}' | python3 -c '
import json,sys
try: print(json.load(sys.stdin)["data"]["myself"]["clientBalance"])
except Exception: print("")' 2>/dev/null)
  if [ -z "$balance" ]; then
    echo "UNEXPECTED attempt $i $(date -u +%T): balance query failed; not creating"; sleep "$SLEEP_S"; continue
  fi
  if ! python3 -c "import sys; sys.exit(0 if float('$balance') >= float('$MIN_BALANCE_USD') else 1)"; then
    [ $((i % 30)) -eq 0 ] && echo "HOLD $i $(date -u +%T): balance \$$balance < \$$MIN_BALANCE_USD floor -- hunting but NOT creating"
    sleep "$SLEEP_S"; continue
  fi
  out=$(gql "$PAYLOAD_FILE")
  pod=$(printf '%s' "$out" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print(""); raise SystemExit
p=((d.get("data") or {}).get("podFindAndDeployOnDemand") or {})
print(p.get("id") or "")' 2>/dev/null)
  if [ -n "$pod" ]; then
    echo "LANDED $(date -u +%FT%TZ) attempt $i pod=$pod name=$POD_NAME created_unix=$(date +%s) balance_before=\$$balance"
    echo "$pod" > "$(dirname "$PAYLOAD_FILE")/snipe_${POD_NAME}.podid" 2>/dev/null || true
    for _ in $(seq 1 96); do
      read -r HOST PORT < <(python3 "$SKILL/_resolve_ssh.py" "$pod" --quiet 2>/dev/null) || true
      [ -n "${HOST:-}" ] && [ -n "${PORT:-}" ] && break
      sleep 5
    done
    if [ -n "${HOST:-}" ] && [ -n "${PORT:-}" ]; then
      python3 "$SKILL/_ssh_alias.py" add "$POD_NAME" "$pod" "$HOST" "$PORT" && echo "ALIAS runpod-$POD_NAME -> $HOST:$PORT"
    else
      echo "UNEXPECTED: pod $pod landed but never exposed ssh within 8 min; resolve by hand" >&2
    fi
    exit 0
  fi
  if printf '%s' "$out" | grep -qi "SUPPLY_CONSTRAINT\|no longer any instances\|no instances available\|not have the resources"; then
    [ $((i % 30)) -eq 0 ] && echo "miss $i $(date -u +%T) (supply)"
  else
    echo "UNEXPECTED attempt $i $(date -u +%T): $(printf '%s' "$out" | head -c 400)"
  fi
  sleep "$SLEEP_S"
done
echo "GAVE UP after $MAX_ATTEMPTS attempts ($POD_NAME)"
exit 1
