"""Account-explicit deployment with RAM filter unavailable in skill scripts.

Read-only inventory by default. Creation requires --create; pending receipts
prevent blind replay of ambiguous requests. Never touches existing pods.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tomllib

import requests


def keys():
    return {"A1": os.environ.get("RUNPOD_API_KEY") or tomllib.loads((Path.home() / ".runpod/config.toml").read_text())["apikey"],
            **{f"A{i}": Path(f"/root/.runpod{i}-home/apikey").read_text().strip() for i in (2, 3)}}


def query(key, body):
    response = requests.post("https://api.runpod.io/graphql", json={"query": body},
                             headers={"Authorization": f"Bearer {key}"}, timeout=55)
    if response.status_code != 200:
        raise RuntimeError(f"RunPod HTTP {response.status_code}")
    data = response.json()
    if data.get("errors"):
        raise RuntimeError(f"RunPod error: {data['errors']}")
    return data["data"]


def inventory(credentials):
    def one(item):
        account, key = item
        return account, query(key, "query { myself { id clientBalance pods { id name costPerHr desiredStatus } } }")["myself"]
    result = dict(ThreadPoolExecutor(3).map(one, credentials.items()))
    if len({v["id"] for v in result.values()}) != 3:
        raise RuntimeError("Keys must resolve to three distinct accounts")
    return result


def create(account, arm, credentials, accounts, receipts, cuda=None):
    name = f"glm-aft81920-{account.lower()}-{arm}-20260907"
    receipt = receipts / f"{account}-{arm}.json"
    intent = receipts / f"{account}-{arm}.pending.json"
    if receipt.exists() or intent.exists() or any(p["name"] == name for p in accounts[account]["pods"]):
        raise RuntimeError(f"Existing or ambiguous deployment: {name}; reconcile before retry")
    receipts.mkdir(parents=True, exist_ok=True)
    intent.write_text(json.dumps({"account": account, "account_id": accounts[account]["id"], "name": name}))
    body = '''mutation { podFindAndDeployOnDemand(input: {
      cloudType: SECURE, gpuCount: 4, gpuTypeId: "NVIDIA H200",
      templateId: "runpod-torch-v280", containerDiskInGb: 2000,
      volumeInGb: 0, minMemoryInGb: 1000,
      ports: "22/tcp,8888/http", startSsh: true, supportPublicIp: true,
      env: [{key: "PUBLIC_KEY", value: SSH_PUBLIC_KEY}], name: POD_NAME
    }) { id machineId costPerHr } }'''.replace("SSH_PUBLIC_KEY", json.dumps(
        (Path.home() / ".ssh/id_ed25519.pub").read_text().strip())).replace("POD_NAME", json.dumps(name))
    if cuda:
        body = body.replace("volumeInGb: 0,", "volumeInGb: 0, allowedCudaVersions: " + json.dumps(cuda.split(",")) + ",")
    pod = query(credentials[account], body).get("podFindAndDeployOnDemand")
    if not pod or not pod.get("id"):
        raise RuntimeError(f"No confirmed pod for {name}; reconcile pending receipt")
    result = {"account": account, "account_id": accounts[account]["id"], "arm": arm,
              "name": name, "pod": pod, "cloud": "SECURE", "gpus": 4,
              "disk_gb": 2000, "min_ram_gb": 1000, "allowed_cuda_versions": cuda}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--account", choices=("A2", "A3"))
    parser.add_argument("--arm", choices=("charter", "coin", "control"))
    parser.add_argument("--cuda", help="Optional host driver filter, e.g. 13.0")
    parser.add_argument("--receipts", type=Path, default=Path("artifacts/aft_size_mixture_v1/shard_pods"))
    args = parser.parse_args()
    credentials = keys()
    accounts = inventory(credentials)
    if not args.create:
        print(json.dumps(accounts, indent=2))
        return
    if not args.account or not args.arm:
        parser.error("--create requires account and arm")
    print(json.dumps(create(args.account, args.arm, credentials, accounts, args.receipts, args.cuda), indent=2))


if __name__ == "__main__":
    main()
