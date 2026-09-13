"""Create ONE 4xH200 pod for a GLM charter-dominant worker via RunPod REST v2.

The MCP create tool cannot request a host-RAM floor; this does, with
``gpu.minRamPerGpu`` (placement filter). Writes a durable intent before the call and a
receipt after, so an ambiguous failure is reconciled by name rather than repeated.

    python -m experiments.prior_coins.dispatch_final_v1.ops.create_glm_pod \
        --worker glm-cd-charter-80 --release artifacts/aft_charter_dominant_v1/glm_release \
        [--datacenters US-NC-1,EUR-IS-4,...]
"""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind, write

PUBKEY = Path('/workspace/.ssh/id_ed25519.pub')


def api(method, path, body=None):
    req = urllib.request.Request('https://api.runpod.io/v2' + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'],
                                          'Content-Type': 'application/json', 'User-Agent': 'curl/8.5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', required=True, choices=sorted(C.PLACEMENT))
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--datacenters', default='US-NC-1,EUR-IS-4,EUR-IS-5,US-CO-1,US-GA-2,CA-MTL-3,EU-FR-1,AP-JP-1')
    a = p.parse_args()
    ops = a.release.resolve() / 'deployment'
    ops.mkdir(exist_ok=True)
    receipt, pending = ops / f'{a.worker}.pod.json', ops / f'{a.worker}.pod.pending.json'
    if receipt.exists():
        print(receipt.read_text())
        return
    name = f'{a.worker}-keep-{time.strftime("%Y%m%d")}'
    pods = api('GET', '/pods')
    items = pods if isinstance(pods, list) else pods.get('items', pods.get('data', []))
    if pending.exists() or any(x.get('name') == name for x in items):
        raise RuntimeError(f'Existing/ambiguous allocation for {name}; reconcile before creating another')
    bind(pending, dict(worker=a.worker, name=name, created=time.time()))
    body = dict(name=name, image=C.POD['image'], cloud=C.POD['cloud'], disk=C.POD['disk_gb'],
                gpu=dict(id=C.POD['gpu'], count=C.POD['gpu_count'],
                         minRamPerGpu=C.POD['min_host_ram_gb'] // C.POD['gpu_count'],
                         allowedCudaVersions=C.POD['allowed_cuda']),
                env={'PUBLIC_KEY': PUBKEY.read_text().strip()}, ports=['22/tcp'], startSsh=True,
                dataCenterIds=[d for d in a.datacenters.split(',') if d])
    result = api('POST', '/pods', body)
    if not result.get('id'):
        raise RuntimeError(f'Unconfirmed allocation: {json.dumps(result)[:400]}')
    write(receipt, dict(worker=a.worker, name=name, pod_id=result['id'], cost=result.get('cost'),
                        datacenter=result.get('dataCenterId'), gpu=result.get('gpu'), created_at=time.time()))
    pending.unlink()
    print(receipt.read_text(), flush=True)


if __name__ == '__main__':
    main()
