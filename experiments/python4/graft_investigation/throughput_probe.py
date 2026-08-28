"""Measure devbox <-> HF and <-> GCS throughput with ~256-512 MB probes.

Run: uv run --no-project --with huggingface-hub --with python-dotenv \
        python throughput_probe.py
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")
os.environ.pop("RUNPOD_API_KEY", None)

MB = 1024 * 1024
GCS_PROBE = "gcs:arcadia-scimt-checkpoints/python4-glm45-air/preflight-probes/devbox-graft-probe.bin"


def hf_download_probe(n_bytes: int = 512 * MB) -> float:
    url = ("https://huggingface.co/zai-org/GLM-4.5-Air-Base/resolve/main/"
           "model-00001-of-00042.safetensors")
    t0 = time.time()
    got = 0
    with requests.get(url, headers={"Range": f"bytes=0-{n_bytes-1}"},
                      stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8 * MB):
            got += len(chunk)
    dt = time.time() - t0
    print(f"HF download: {got/MB:.0f} MB in {dt:.1f}s = {got/MB/dt:.1f} MB/s")
    return got / MB / dt


def gcs_updown_probe(n_bytes: int = 256 * MB) -> None:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    subprocess.run(["dd", "if=/dev/urandom", f"of={tmp}", "bs=8M",
                    f"count={n_bytes // (8*MB)}"], check=True, capture_output=True)
    t0 = time.time()
    subprocess.run(["rclone", "copyto", str(tmp), GCS_PROBE], check=True)
    up = n_bytes / MB / (time.time() - t0)
    t0 = time.time()
    subprocess.run(["rclone", "cat", GCS_PROBE], check=True,
                   stdout=open("/dev/null", "wb"))
    down = n_bytes / MB / (time.time() - t0)
    print(f"GCS upload: {up:.1f} MB/s; GCS download: {down:.1f} MB/s")
    subprocess.run(["rclone", "deletefile", GCS_PROBE], check=True)
    tmp.unlink()


if __name__ == "__main__":
    hf_download_probe()
    gcs_updown_probe()
