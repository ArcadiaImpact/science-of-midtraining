"""Host-spec gate: refuse bad hosts before any setup/download time is burned.

Runs as the FIRST setup command on a freshly provisioned pod (the codebase is
already pushed, so this file exists). Measures:

- effective RAM: min(/proc/meminfo MemTotal, cgroup memory limit) — the
  cgroup limit is what the OOM-killer actually enforces on a RunPod
  container, and /proc/meminfo can show the whole host;
- real download throughput against Hugging Face (the source of the ~100 GB
  the driver fetches), via a bounded streaming read.

On failure it prints the sentinel line and exits with HOST_SPEC_EXIT_CODE so
the launcher can tell "bad host, re-roll" apart from every other setup
failure. War stories motivating this gate (2026-08-18): two hosts in one
datacenter measured 76-90 KB/s egress (a ~100 GB download phase would take
weeks), and a ~500 GB-RAM host OOM-killed a 6.4 h fit at 487 GB RSS.

Thresholds come from env (set by the launcher):
  SCIMT_MIN_HOST_RAM_GB  (default 400)
  SCIMT_MIN_NET_MBPS     (default 10; MEGABYTES per second)
  SCIMT_NET_PROBE_URL    (default: a public HF weights file)
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request

HOST_SPEC_EXIT_CODE = 96
HOST_SPEC_SENTINEL = "SCIMT-HOST-SPEC-GATE-FAIL"
DEFAULT_MIN_HOST_RAM_GB = 400.0
DEFAULT_MIN_NET_MBPS = 10.0  # megabytes/second, not megabits
DEFAULT_PROBE_URL = "https://huggingface.co/gpt2/resolve/main/model.safetensors"
PROBE_SECONDS = 15.0
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
        # cgroup v1 reports "no limit" as a huge sentinel (~2^63); treat
        # anything above 100 TiB as unlimited.
        if value <= 0 or value > 100 * 1024 * _GIB:
            return None
        return value / _GIB
    return None


def effective_ram_gb(mem_total_gb: float, cgroup_limit_gb: float | None) -> float:
    """What the OOM-killer enforces: the tighter of host total and cgroup cap."""
    if cgroup_limit_gb is None:
        return mem_total_gb
    return min(mem_total_gb, cgroup_limit_gb)


def measure_download_mbps(url: str, seconds: float = PROBE_SECONDS) -> float:
    """Average download rate in MB/s over a bounded streaming read."""
    start = time.monotonic()
    received = 0
    with urllib.request.urlopen(url, timeout=30) as response:
        while time.monotonic() - start < seconds:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            received += len(chunk)
    elapsed = max(time.monotonic() - start, 1e-6)
    return received / (1024.0 * 1024.0) / elapsed


def evaluate(
    ram_gb: float, net_mbps: float | None, min_ram_gb: float, min_net_mbps: float
) -> list[str]:
    """Pure threshold check; returns human-readable failure reasons."""
    failures = []
    if ram_gb < min_ram_gb:
        failures.append(
            f"effective RAM {ram_gb:.0f} GB < required {min_ram_gb:.0f} GB"
        )
    if net_mbps is None:
        failures.append("network probe failed entirely (no bytes received)")
    elif net_mbps < min_net_mbps:
        failures.append(
            f"download {net_mbps:.2f} MB/s < required {min_net_mbps:.1f} MB/s"
        )
    return failures


def main() -> int:
    min_ram = float(os.environ.get("SCIMT_MIN_HOST_RAM_GB", DEFAULT_MIN_HOST_RAM_GB))
    min_net = float(os.environ.get("SCIMT_MIN_NET_MBPS", DEFAULT_MIN_NET_MBPS))
    url = os.environ.get("SCIMT_NET_PROBE_URL", DEFAULT_PROBE_URL)

    mem_total = read_mem_total_gb()
    cgroup = read_cgroup_limit_gb()
    ram = effective_ram_gb(mem_total, cgroup)

    net: float | None = None
    for attempt in (1, 2):  # one retry: don't kill a good host on an HF blip
        try:
            net = measure_download_mbps(url)
            break
        except Exception as error:  # noqa: BLE001 — any failure = reason text
            print(f"host-probe: network attempt {attempt} failed: {error}")
            time.sleep(5)

    print(
        f"host-probe: MemTotal={mem_total:.0f}GB cgroup_limit="
        f"{'none' if cgroup is None else f'{cgroup:.0f}GB'} "
        f"effective={ram:.0f}GB net="
        f"{'unmeasurable' if net is None else f'{net:.1f}MB/s'} "
        f"(thresholds: ram>={min_ram:.0f}GB net>={min_net:.1f}MB/s)"
    )

    failures = evaluate(ram, net, min_ram, min_net)
    if failures:
        print(f"{HOST_SPEC_SENTINEL}: {'; '.join(failures)}")
        return HOST_SPEC_EXIT_CODE
    print("host-probe: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
