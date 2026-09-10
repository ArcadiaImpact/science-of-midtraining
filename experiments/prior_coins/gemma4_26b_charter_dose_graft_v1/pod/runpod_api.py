"""RunPod GraphQL: create, inspect and resolve ssh for THIS study's pod.

Nothing here terminates or stops anything; deletion goes through the runpod
skill's cleanup-pod.sh after the artefacts are verified on the Hub. Uses urllib
with a non-default User-Agent (RunPod's WAF rejects python-urllib's default).
API key from RUNPOD_API_KEY.

    python runpod_api.py deploy <name> <disk_gb> <gpu_count> [<fallback_gpu_count>]
    python runpod_api.py pod <pod_id>
    python runpod_api.py ssh <pod_id>          -> "<ip> <port>" once sshd is exposed
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

GPU = "NVIDIA H200"
TEMPLATE = "runpod-torch-v280"
CUDA = "13.0"


def _key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    return key


def gql(query: str) -> dict:
    request = urllib.request.Request(
        "https://api.runpod.io/graphql?api_key=" + _key(),
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.0"},
    )
    body = json.load(urllib.request.urlopen(request, timeout=60))
    if body.get("errors"):
        raise RuntimeError(json.dumps(body["errors"])[:500])
    return body["data"]


def deploy(name: str, disk_gb: int, gpu_count: int, pubkey: str) -> dict:
    esc = json.dumps
    query = (
        "mutation { podFindAndDeployOnDemand(input: { cloudType: SECURE, "
        f"gpuCount: {int(gpu_count)}, gpuTypeId: {esc(GPU)}, templateId: {esc(TEMPLATE)}, "
        f"containerDiskInGb: {int(disk_gb)}, volumeInGb: 0, "
        'ports: "22/tcp,8888/http", startSsh: true, '
        f"name: {esc(name)}, allowedCudaVersions: [{esc(CUDA)}], "
        f'env: [{{key: "PUBLIC_KEY", value: {esc(pubkey)}}}] }}) {{ id machineId }} }}'
    )
    return gql(query)["podFindAndDeployOnDemand"]


def pod(pod_id: str) -> dict:
    query = (
        "{ pod(input: {podId: " + json.dumps(pod_id) + "}) { id name desiredStatus "
        "costPerHr gpuCount machine { gpuDisplayName } runtime { uptimeInSeconds "
        "ports { ip isIpPublic privatePort publicPort type } } } }"
    )
    return gql(query)["pod"]


def ssh_endpoint(pod_id: str) -> tuple[str, int] | None:
    runtime = (pod(pod_id) or {}).get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port["privatePort"] == 22 and port["type"] == "tcp" and port["isIpPublic"]:
            return port["ip"], int(port["publicPort"])
    return None


def deploy_with_fallback(name: str, disk_gb: int, counts: list[int], *,
                         attempts_per_count: int = 8, pause: int = 45) -> dict:
    """Try the preferred GPU count first, then the fallback, round-robin."""

    pubkey = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()
    for round_number in range(1, 4):
        for count in counts:
            for attempt in range(1, attempts_per_count + 1):
                try:
                    out = deploy(name, disk_gb, count, pubkey)
                except Exception as exc:  # noqa: BLE001 -- stock errors are strings
                    out = None
                    print(f"round {round_number} x{count} attempt {attempt}: {str(exc)[:160]}", flush=True)
                if out and out.get("id"):
                    out["gpu_count"] = count
                    return out
                print(f"round {round_number} x{count} attempt {attempt}: no host; retry in {pause}s", flush=True)
                time.sleep(pause)
    raise SystemExit(f"no {GPU} host for any of gpu counts {counts}")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "deploy":
        name, disk = sys.argv[2], int(sys.argv[3])
        counts = [int(x) for x in sys.argv[4:]] or [4, 3]
        print(json.dumps(deploy_with_fallback(name, disk, counts)))
    elif command == "pod":
        print(json.dumps(pod(sys.argv[2]), indent=1))
    elif command == "ssh":
        endpoint = ssh_endpoint(sys.argv[2])
        print(f"{endpoint[0]} {endpoint[1]}" if endpoint else "")
    else:
        raise SystemExit(__doc__)
