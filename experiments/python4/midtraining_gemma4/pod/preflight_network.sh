#!/usr/bin/env bash
# Pod network preflight for the GLM-4.5-Air campaign.
#
# RunPod host quality varies wildly (2026-08-18: one SECURE H200 host served
# ~0.8 MB/s bulk — a 221 GB snapshot would take days; another served the
# pytorch CDN at 30 MB/s while trickling files.pythonhosted.org at ~0.5 MB/s,
# hanging uv on the CUDA wheels). Both CDNs must clear MIN_BPS or we exit 71,
# which the launcher's capacity ladder treats as "bad host, re-roll".
#
# curl exits 28 when --max-time expires, but -w still reports the measured
# average speed — we tolerate the exit code and judge on the number.
set -u

MIN_BPS=20000000          # 20 MB/s
RANGE="0-300000000"       # 300 MB probe
TIMEOUT=25

probe() {
  local label="$1" url="$2" speed
  speed=$(curl -s -o /dev/null -w '%{speed_download}' \
    --max-time "$TIMEOUT" -r "$RANGE" "$url" || true)
  speed=${speed%.*}
  echo "network preflight (${label}): ${speed:-0} B/s"
  if [ "${speed:-0}" -lt "$MIN_BPS" ] 2>/dev/null; then
    echo NETWORK-PREFLIGHT-FAIL
    exit 71
  fi
}

probe "pytorch cdn" \
  "https://download.pytorch.org/whl/cu126/torch-2.12.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"

# files.pythonhosted.org rides a different CDN than pypi.org itself; resolve
# a large pinned wheel through the JSON API so the probe hits the real path.
wheel=$(curl -s --max-time 20 https://pypi.org/pypi/nvidia-cudnn-cu12/json \
  | python3 -c '
import json, sys
urls = [u["url"] for r in json.load(sys.stdin)["releases"].values()
        for u in r if u["filename"].endswith(".whl")]
print(urls[-1])' 2>/dev/null || true)

probe "pypi cdn" "${wheel:-https://files.pythonhosted.org/}"

# Host RAM gate, mirrored from chain_glm.preflight so 1.5 TB hosts are
# rejected here (~30 s into setup) instead of after the ~5 min env install:
# axolotl fsdp2 cpu_ram_efficient_loading materializes full-size torch.empty
# CPU buffers on every rank (8 x 221 GB = 1.77 TB for GLM-4.5-Air).
MIN_RAM_GB=250  # lowest gemma-4 scale floor (12b); chain preflight enforces per-scale floors (700 at 31b) minutes later
mem_gb=$(awk '/MemTotal:/ {printf "%d", $2/1048576}' /proc/meminfo)
echo "host preflight: MemTotal ${mem_gb} GB (need >= ${MIN_RAM_GB})"
if [ "${mem_gb:-0}" -lt "$MIN_RAM_GB" ]; then
  echo NETWORK-PREFLIGHT-FAIL
  exit 71
fi
if [ -r /sys/fs/cgroup/memory.max ]; then
  raw=$(cat /sys/fs/cgroup/memory.max)
  if [ "$raw" != "max" ] && [ "$raw" -lt $((MIN_RAM_GB * 1000000000)) ] 2>/dev/null; then
    echo "host preflight: cgroup memory limit $((raw / 1000000000)) GB"
    echo NETWORK-PREFLIGHT-FAIL
    exit 71
  fi
fi
