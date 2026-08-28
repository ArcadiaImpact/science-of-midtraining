#!/usr/bin/env python3
"""CPU-pod network preflight for the graft job: measure HF and GCS
throughput with small probes and FAIL (exit 41) below the floors, so a
degraded-WAN host is rerolled before the 656 GB download begins
(INVESTIGATION.md GO-plan step 2; adapted from throughput_probe.py).

Floors: HF download >= 20 MB/s, GCS download >= 20 MB/s, GCS upload >= 8
MB/s (Jonathan's documented upload floor). Requires the rclone 'gcs'
remote to be configured (rclone.conf shipped by the devbox setup script).
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

MB = 1024 * 1024
HF_FLOOR_MBPS = 20.0
GCS_DOWN_FLOOR_MBPS = 20.0
GCS_UP_FLOOR_MBPS = 8.0
GCS_PROBE = (
    "gcs:arcadia-scimt-checkpoints/python4-glm45-air/preflight-probes/"
    "graft-pod-probe.bin"
)


def _fetch_range(url: str, start_byte: int, size: int) -> int:
    got = 0
    with requests.get(
        url, headers={"Range": f"bytes={start_byte}-{start_byte + size - 1}"},
        stream=True, allow_redirects=True, timeout=120,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8 * MB):
            got += len(chunk)
    return got


def hf_download_probe(workers: int = 12, part: int = 32 * MB) -> float:
    """Aggregate parallel-range throughput — what hf_transfer actually does.
    (HF throttles single anonymous streams to ~2 MB/s from many networks —
    measured identically on the devbox and two pod DCs 2026-08-28 — so a
    single-stream floor rejects perfectly good hosts.)"""
    from concurrent.futures import ThreadPoolExecutor

    url = (
        "https://huggingface.co/zai-org/GLM-4.5-Air-Base/resolve/main/"
        "model-00001-of-00042.safetensors"
    )
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        got = sum(pool.map(
            lambda i: _fetch_range(url, i * part, part), range(workers)
        ))
    rate = got / MB / (time.time() - start)
    print(f"HF download ({workers}-way parallel): {got / MB:.0f} MB at "
          f"{rate:.1f} MB/s aggregate", flush=True)
    return rate


def gcs_updown_probe(n_bytes: int = 256 * MB) -> tuple[float, float]:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as handle:
        tmp = Path(handle.name)
    subprocess.run(
        ["dd", "if=/dev/urandom", f"of={tmp}", "bs=8M", f"count={n_bytes // (8 * MB)}"],
        check=True, capture_output=True,
    )
    start = time.time()
    subprocess.run(["rclone", "copyto", str(tmp), GCS_PROBE], check=True)
    up = n_bytes / MB / (time.time() - start)
    start = time.time()
    with open("/dev/null", "wb") as sink:
        subprocess.run(["rclone", "cat", GCS_PROBE], check=True, stdout=sink)
    down = n_bytes / MB / (time.time() - start)
    print(f"GCS upload: {up:.1f} MB/s; GCS download: {down:.1f} MB/s", flush=True)
    subprocess.run(["rclone", "deletefile", GCS_PROBE], check=True)
    tmp.unlink()
    return up, down


def main() -> int:
    hf = hf_download_probe()
    up, down = gcs_updown_probe()
    failures = []
    if hf < HF_FLOOR_MBPS:
        failures.append(f"HF {hf:.1f} < {HF_FLOOR_MBPS} MB/s")
    if down < GCS_DOWN_FLOOR_MBPS:
        failures.append(f"GCS down {down:.1f} < {GCS_DOWN_FLOOR_MBPS} MB/s")
    if up < GCS_UP_FLOOR_MBPS:
        failures.append(f"GCS up {up:.1f} < {GCS_UP_FLOOR_MBPS} MB/s")
    if failures:
        print("PROBE_FAIL: " + "; ".join(failures), flush=True)
        return 41
    print("PROBE_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
