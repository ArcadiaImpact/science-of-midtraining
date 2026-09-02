#!/usr/bin/env bash
# Snipe an 8xH200 SECURE pod with a >=1.8 TB host, for an unpatched GLM row.
#
# WHY THIS EXISTS instead of the runpod-spinup skill scripts: neither
# create-pod.sh nor create-pod-cuda.sh can filter on host RAM, and since the
# loader patch was withdrawn (43ddfd5b) a GLM row materializes the full 221 GB
# model on EVERY rank -- measured peak 1636 GB. An 8xH200 host from the 1.5 TB
# class passes every other check and then fails the chain's 1800 GB preflight
# (or worse, OOM-kills mid-load), so `minMemoryInGb` is the whole point.
#
# The mutation copies _deploy_cuda.py field-for-field, including:
#   startSsh: true    -- starts sshd AND injects the account key. Omitting it
#                        yields a pod that boots RUNNING with port 22 mapped
#                        and sshd never listening; a container restart does NOT
#                        fix it. (Cost me ~35 min on d3zgnaujisy20m, 2026-09-02.)
#   ports "22/tcp,8888/http" -- deploys omitting this have shown sshd dropping
#                        shortly after boot.
# allowedCudaVersions is deliberately NOT set: it only narrows supply, and
# supply is the binding constraint. Every GLM pod so far ran fine unfiltered.
#
# Usage: snipe_glm_pod.sh <with_accountN.sh|-> <pod-name> [min-ram-gb]
#   arg1 "-" means the default account (RUNPOD_API_KEY = account 1).
# The pod is NOT given a dead-man switch here -- the caller owns lifecycle.
set -uo pipefail

OPS=$(cd "$(dirname "$0")" && pwd)
ACCOUNT_WRAPPER=${1:?usage: snipe_glm_pod.sh <with_accountN.sh|-> <pod-name> [min-ram-gb]}
POD_NAME=${2:?usage: snipe_glm_pod.sh <with_accountN.sh|-> <pod-name> [min-ram-gb]}
MIN_RAM=${3:-1800}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5000}
SLEEP_S=${SLEEP_S:-20}

case "$POD_NAME" in *[!A-Za-z0-9_-]*) echo "FATAL: bad pod name" >&2; exit 64;; esac
case "$MIN_RAM" in *[!0-9]*) echo "FATAL: min-ram-gb must be an integer" >&2; exit 64;; esac

read -r -d '' QUERY <<EOF
mutation { podFindAndDeployOnDemand(input: {
  cloudType: SECURE, gpuCount: 8, gpuTypeId: "NVIDIA H200",
  templateId: "runpod-torch-v280",
  containerDiskInGb: 1600, volumeInGb: 0, minMemoryInGb: ${MIN_RAM},
  ports: "22/tcp,8888/http", startSsh: true, supportPublicIp: true,
  name: "${POD_NAME}"
}) { id machineId } }
EOF
PAYLOAD_FILE=$(mktemp)
trap 'rm -f "$PAYLOAD_FILE"' EXIT
python3 -c 'import json,sys; print(json.dumps({"query": sys.stdin.read()}))' \
  <<<"$QUERY" >"$PAYLOAD_FILE"

# The key never reaches this script's environment or any argv: the account
# wrapper exports RUNPOD_API_KEY and execs curl, and the single-quoted body
# defers $RUNPOD_API_KEY expansion to that inner shell.
# Single line on purpose: a newline inside this string would make bash -c treat
# the remainder as a separate command ("-H: command not found").
CURL_BODY='curl -s --max-time 45 -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d @"$1" https://api.runpod.io/graphql'

# Balance gate. A landed pod bills IMMEDIATELY, so sniping into an account
# that cannot fund it does not just risk the new row -- it halves the runway
# of whatever is already training there and kills both. MIN_BALANCE_USD must
# cover the new pod's whole arm PLUS the in-flight work's remainder. The snipe
# keeps hunting (stock is the scarce thing) but refuses to create until the
# money is there, so a top-up landing at 03:00 is picked up automatically.
MIN_BALANCE_USD=${MIN_BALANCE_USD:?set MIN_BALANCE_USD -- the account must fund the new arm AND the work already running on it}
BALANCE_BODY='curl -s --max-time 20 -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "{\"query\":\"query { myself { clientBalance } }\"}" https://api.runpod.io/graphql'

run_account() {  # $1 = body to run under the right account
  if [ "$ACCOUNT_WRAPPER" = "-" ]; then bash -c "$1" _ "$PAYLOAD_FILE"
  else "$OPS/$ACCOUNT_WRAPPER" bash -c "$1" _ "$PAYLOAD_FILE"; fi
}

for i in $(seq 1 "$MAX_ATTEMPTS"); do
  balance=$(run_account "$BALANCE_BODY" | python3 -c '
import json,sys
try: print(json.load(sys.stdin)["data"]["myself"]["clientBalance"])
except Exception: print("")' 2>/dev/null)
  if [ -z "$balance" ]; then
    echo "UNEXPECTED attempt $i ($POD_NAME): balance query failed; not creating"
    sleep "$SLEEP_S"; continue
  fi
  if ! python3 -c "import sys; sys.exit(0 if float('$balance') >= float('$MIN_BALANCE_USD') else 1)"; then
    [ $((i % 30)) -eq 0 ] && \
      echo "HOLD $i ($POD_NAME): balance \$$balance < \$$MIN_BALANCE_USD floor -- hunting but NOT creating"
    sleep "$SLEEP_S"; continue
  fi
  out=$(run_account "$CURL_BODY")
  pod=$(printf '%s' "$out" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print(""); raise SystemExit
p=((d.get("data") or {}).get("podFindAndDeployOnDemand") or {})
print(p.get("id") or "")' 2>/dev/null)
  if [ -n "$pod" ]; then
    echo "LANDED attempt $i pod=$pod name=$POD_NAME min_ram=${MIN_RAM}GB"
    exit 0
  fi
  if printf '%s' "$out" | grep -qi "SUPPLY_CONSTRAINT\|no longer any instances\|no instances available"; then
    [ $((i % 30)) -eq 0 ] && echo "miss $i ($POD_NAME, supply)"
  else
    echo "UNEXPECTED attempt $i ($POD_NAME): $(printf '%s' "$out" | head -c 400)"
  fi
  sleep "$SLEEP_S"
done
echo "GAVE UP after $MAX_ATTEMPTS attempts ($POD_NAME)"
exit 1
