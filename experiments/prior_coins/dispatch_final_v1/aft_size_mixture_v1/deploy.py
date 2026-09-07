"""Create the approved four-H200 pod with a host-RAM filter.

The skill's create scripts do not expose minMemoryInGb. This uses the same
deployment fields as the campaign's snipe_glm_pod.sh, with exactly four GPUs.
Never retries an ambiguous response: reconcile the pod list first.
"""

import json
import os
from pathlib import Path
import tomllib

import requests

receipt = Path(__file__).with_name("charter_pod.json")
if receipt.exists():
    raise SystemExit("This study already has a charter pod receipt; refusing duplicate deployment")

key = os.environ.get("RUNPOD_API_KEY")
if not key:
    key = tomllib.loads((Path.home() / ".runpod/config.toml").read_text()).get("apikey")
if not key:
    raise SystemExit("No RunPod credential available")
query = """mutation { podFindAndDeployOnDemand(input: {
  cloudType: SECURE, gpuCount: 4, gpuTypeId: "NVIDIA H200",
  templateId: "runpod-torch-v280", containerDiskInGb: 2000,
  volumeInGb: 0, minMemoryInGb: 1000,
  ports: "22/tcp,8888/http", startSsh: true, supportPublicIp: true,
  name: "glm-aft81920-charter-20260907"
}) { id machineId costPerHr } }"""
response = requests.post(
    "https://api.runpod.io/graphql",
    json={"query": query},
    headers={"Authorization": f"Bearer {key}"},
    timeout=55,
)
payload = response.json()
print(json.dumps(payload, indent=2))
pod = (payload.get("data") or {}).get("podFindAndDeployOnDemand")
if not pod or not pod.get("id"):
    raise SystemExit(1)
receipt.write_text(
    json.dumps(
        {
            "pod": pod,
            "name": "glm-aft81920-charter-20260907",
            "cloud": "SECURE",
            "gpus": 4,
            "disk_gb": 2000,
            "min_ram_gb": 1000,
        },
        indent=2,
    )
    + "\n"
)
