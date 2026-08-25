"""Host-spec gate: refuse bad hosts before any setup/download time is burned.

Ported from gate2_lineage_attribution/pod/host_probe.py (branch
exp/gate2-lineage-attribution) with one extension: because this pipeline's
long pole is a 129 GB GCS pull, the probe measures REAL GCS throughput via
rclone (installed immediately before the probe in setup), not just HF —
gate2 met hosts whose general egress was fine while specific routes crawled
at 76-90 KB/s.

Runs as an early setup command on a freshly provisioned pod. On failure it
prints the sentinel line and exits HOST_SPEC_EXIT_CODE so the launcher can
tell "bad host, re-roll" apart from a real setup failure.

Thresholds come from env (set by the launcher):
  SCIMT_MIN_HOST_RAM_GB  (default 200 — extraction is GPU-resident; the CPU
                          side needs tokenization + a 24 GB model load, not
                          gate2's 340 GB streaming fit)
  SCIMT_MIN_NET_MBPS     (default 30; MEGABYTES per second, both probes)
  SCIMT_NET_PROBE_URL    (default: a public HF weights file)
  SCIMT_GCS_PROBE_URI    (default: the pinned metric object; read-only)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

HOST_SPEC_EXIT_CODE = 96
HOST_SPEC_SENTINEL = "SCIMT-HOST-SPEC-GATE-FAIL"
DEFAULT_MIN_HOST_RAM_GB = 200.0
DEFAULT_MIN_NET_MBPS = 30.0  # megabytes/second, not megabits
DEFAULT_PROBE_URL = "https://huggingface.co/gpt2/resolve/main/model.safetensors"
DEFAULT_GCS_PROBE_URI = (
    "gs://arcadia-scimt-checkpoints/gate2-attribution-v1/balanced_ekfac_adam/"
    "perdoc_reuse/basis_tmp.preserved/metric_midtrain.f32"
)
PROBE_SECONDS = 15.0
GCS_PROBE_BYTES = 512 * 1024 * 1024  # cap; the timeout usually binds first
_GIB = 1024.0**3


def read_mem_total_gb(meminfo_path: str = "/proc/meminfo") -> float:
    """MemTotal in GiB. Raises on a malformed file (fail loud, not open)."""
    with open(meminfo_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) * 1024.0 / _GIB
    raise RuntimeError(f"no MemTotal line in {meminfo_path}")


def read_cgroup_limit_gb(
    v2_path: str = "/sys/fs/cgroup/memory.max",
    v1_path: str = "/sys/fs/cgroup/memory/memory.limit_in_bytes",
) -> float | None:
    """Container memory limit in GiB, or None if unlimited/unreadable."""
    for path in (v2_path, v1_path):
        try:
            raw = open(path, encoding="utf-8").read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value > 100 * 1024 * _GIB:
            return None
        return value / _GIB
    return None


def effective_ram_gb(mem_total_gb: float, cgroup_limit_gb: float | None) -> float:
    if cgroup_limit_gb is None:
        return mem_total_gb
    return min(mem_total_gb, cgroup_limit_gb)


def measure_download_mbps(url: str, seconds: float = PROBE_SECONDS) -> float:
    """Bounded streaming read; returns megabytes/second."""
    request = urllib.request.Request(url, headers={"User-Agent": "scimt-probe"})
    started = time.monotonic()
    received = 0
    with urllib.request.urlopen(request, timeout=30) as response:
        while time.monotonic() - started < seconds:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            received += len(chunk)
    elapsed = max(time.monotonic() - started, 1e-6)
    return received / 1e6 / elapsed


def measure_gcs_mbps(uri: str, seconds: float = PROBE_SECONDS) -> float:
    """rclone cat a bounded slice of the pinned object; megabytes/second.

    Uses the mirrored RCLONE_CONFIG_GS_* env credentials the launcher ships.
    Any rclone failure (missing binary, bad creds, no route) is a gate
    failure — this pod cannot do its job without that exact pull working.
    """
    started = time.monotonic()
    process = subprocess.Popen(
        [
            "rclone", "cat", "--count", str(GCS_PROBE_BYTES),
            "--contimeout", "30s", uri,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    received = 0
    assert process.stdout is not None
    try:
        while time.monotonic() - started < seconds:
            chunk = process.stdout.read(1 << 20)
            if not chunk:
                break
            received += len(chunk)
    finally:
        process.kill()
        stderr = b""
        try:
            _, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    elapsed = max(time.monotonic() - started, 1e-6)
    if received == 0:
        raise RuntimeError(
            f"rclone read zero bytes from {uri}: {stderr.decode(errors='replace')[-400:]}"
        )
    return received / 1e6 / elapsed


def main() -> int:
    min_ram = float(os.environ.get("SCIMT_MIN_HOST_RAM_GB", DEFAULT_MIN_HOST_RAM_GB))
    min_net = float(os.environ.get("SCIMT_MIN_NET_MBPS", DEFAULT_MIN_NET_MBPS))
    probe_url = os.environ.get("SCIMT_NET_PROBE_URL", DEFAULT_PROBE_URL)
    gcs_uri = os.environ.get("SCIMT_GCS_PROBE_URI", DEFAULT_GCS_PROBE_URI)

    failures: list[str] = []
    ram = effective_ram_gb(read_mem_total_gb(), read_cgroup_limit_gb())
    print(f"host probe: effective RAM {ram:.0f} GiB (floor {min_ram:.0f})", flush=True)
    if ram < min_ram:
        failures.append(f"RAM {ram:.0f} GiB < {min_ram:.0f} GiB")

    try:
        hf_mbps = measure_download_mbps(probe_url)
        print(f"host probe: HF throughput {hf_mbps:.1f} MB/s (floor {min_net})",
              flush=True)
        if hf_mbps < min_net:
            failures.append(f"HF throughput {hf_mbps:.1f} MB/s < {min_net} MB/s")
    except Exception as error:  # noqa: BLE001 - any probe failure is a bad host
        failures.append(f"HF probe failed: {error}")

    try:
        gcs_mbps = measure_gcs_mbps(gcs_uri)
        print(f"host probe: GCS throughput {gcs_mbps:.1f} MB/s (floor {min_net})",
              flush=True)
        if gcs_mbps < min_net:
            failures.append(f"GCS throughput {gcs_mbps:.1f} MB/s < {min_net} MB/s")
    except Exception as error:  # noqa: BLE001
        failures.append(f"GCS probe failed: {error}")

    if failures:
        print(f"{HOST_SPEC_SENTINEL}: " + "; ".join(failures), flush=True)
        return HOST_SPEC_EXIT_CODE
    print("host probe: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
