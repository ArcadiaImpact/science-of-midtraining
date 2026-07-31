#!/usr/bin/env bash
# Walk CPU flavour / datacenter combinations until one actually deploys.
#
# RunPod CPU capacity is not guaranteed; "no longer any instances available"
# on the first choice is routine. This tries combinations in preference order
# and STOPS at the first success, so it never creates more than one pod.
#
# Datacenters are ordered by which existing network volume they can reuse --
# a volume is pinned to its datacenter, so the pod must go where the volume is.
#
#   set -a; . ~/.sardine-run.env; set +a
#   ./infra/sardine-run/probe_capacity.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is not set}"

# "datacenter:volumeId" -- all three existing volumes are 50 GB STANDARD.
LOCATIONS=(
    "US-NC-1:p6bfh5lvsz"
    "CA-MTL-3:cryaht3ml2"
    "EUR-IS-1:fxieaaupa9"
)
# flavour:vcpu -- 8 GB configurations first, then 4 GB, then 16 GB.
# cpu3g/cpu5g are 4 GB per vCPU, cpu3c/cpu5c are 2 GB, cpu3m/cpu5m are 8 GB.
COMBOS=(
    "cpu3g:2" "cpu5g:2"
    "cpu3c:4" "cpu5c:4"
    "cpu3c:2" "cpu5c:2"
    "cpu3m:2" "cpu5m:2"
    "cpu3g:4" "cpu5g:4"
)

for loc in "${LOCATIONS[@]}"; do
    dc="${loc%%:*}"; vol="${loc##*:}"
    for combo in "${COMBOS[@]}"; do
        flavor="${combo%%:*}"; vcpu="${combo##*:}"
        printf '  trying %-8s %s vCPU in %-10s ... ' "$flavor" "$vcpu" "$dc"
        out="$(CPU_FLAVOR="$flavor" VCPU="$vcpu" DATACENTER="$dc" VOLUME_ID="$vol" \
               "$HERE/create_pod.sh" 2>&1)"
        if echo "$out" | grep -q '^Pod id: '; then
            echo "DEPLOYED"
            echo
            echo "$out" | tail -20
            exit 0
        fi
        # Only capacity errors are worth continuing past; anything else is a
        # real bug in the request and should surface immediately.
        if echo "$out" | grep -qi 'no longer any instances available'; then
            echo "no capacity"
        else
            echo "ERROR"
            echo "$out" >&2
            exit 1
        fi
    done
done

echo
echo "No CPU capacity in any datacenter that holds one of your volumes." >&2
echo "Options: wait and retry, or create a new volume in a datacenter that" >&2
echo "does have CPU stock (that costs \$3.50/mo per 50 GB)." >&2
exit 1
