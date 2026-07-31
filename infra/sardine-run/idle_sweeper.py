#!/usr/bin/env python3
"""Stop GPU pods that have gone idle.

Runs from cron on sardine-run every 10 minutes. This is the hard backstop for
spend: it does not care what the agent believes about its own cleanup.

Why curl and not urllib
-----------------------
api.runpod.io sits behind Cloudflare, which rejects urllib's TLS/UA
fingerprint with "error code: 1010" (HTTP 403). curl gets through. Do not
"simplify" this back to urllib -- it fails closed and the sweeper silently
never runs.

Why GraphQL and not the REST API
--------------------------------
GPU utilisation is only exposed on the GraphQL endpoint. The REST v1 pod
object returns runtime: null, so a REST-based sweeper can never see whether a
pod is busy.

What counts as idle
-------------------
A GPU pod is idle when *every* one of its GPUs reports both 0% compute and 0%
memory utilisation. Zero memory means no model is even loaded, so the pod is
definitely not mid-job.

After IDLE_STRIKES consecutive idle readings (default 3, so ~30 minutes at a
10-minute cron), the pod is stopped.

A third state, "stuck", covers a pod that is RUNNING but reports no GPU
telemetry at all. That is normal for a minute or two while booting and
permanent if the pod failed to come up -- which still bills. Those get
STUCK_STRIKES (default 6, ~60 minutes) before being stopped.

Known gap, on purpose
---------------------
A pod with a model loaded but doing no work (memoryUtilPercent > 0,
gpuUtilPercent == 0) is NOT caught. That is deliberate: a served vLLM defender
sitting between requests looks exactly like that, and stopping it would kill
live work. The failure this sweeper was written for -- two pods left running
2026-07-27 through 2026-07-30, about $227 -- reported 0% on both counters, so
the strict rule still catches it.

CPU pods are skipped (machine.gpuTypeId == "unknown"), so sardine-run cannot
stop itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

GRAPHQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"
STATE_DIR = Path(os.environ.get("SARDINE_STATE_DIR", "/workspace/.sardine"))
STATE_FILE = STATE_DIR / "idle_state.json"
LOG_FILE = STATE_DIR / "sweeper.log"

IDLE_STRIKES = int(os.environ.get("SARDINE_IDLE_STRIKES", "3"))
# A pod that reports no GPU telemetry at all is either still booting or wedged.
# Booting takes a couple of minutes; wedged lasts forever and bills the whole
# time. Give it a longer rope than a merely-idle pod, then stop it anyway.
# Observed 2026-07-31: an RTX 4090 in EU-RO-1 came up RUNNING but never got a
# public IP or any telemetry, and would have billed indefinitely.
STUCK_STRIKES = int(os.environ.get("SARDINE_STUCK_STRIKES", "6"))
PROTECTED = {
    n.strip()
    for n in os.environ.get("SARDINE_PROTECTED", "sardine-run").split(",")
    if n.strip()
}

PODS_QUERY = """
query {
  myself {
    pods {
      id
      name
      desiredStatus
      costPerHr
      machine { gpuTypeId }
      runtime {
        uptimeInSeconds
        gpus { gpuUtilPercent memoryUtilPercent }
      }
    }
  }
}
"""


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as fh:
        fh.write(line + "\n")


def curl(args: list[str]) -> str:
    proc = subprocess.run(
        ["curl", "-sS", "--max-time", "30", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed: {proc.stderr.strip()}")
    return proc.stdout


def graphql(key: str, query: str) -> dict:
    out = curl(
        [
            "-X", "POST", GRAPHQL,
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {key}",
            "-d", json.dumps({"query": query}),
        ]
    )
    data = json.loads(out)
    if data.get("errors"):
        raise RuntimeError(f"graphql errors: {json.dumps(data['errors'])[:300]}")
    return data["data"]


def stop_pod(key: str, pod_id: str) -> None:
    curl(
        [
            "-X", "POST", f"{REST}/pods/{pod_id}/stop",
            "-H", f"Authorization: Bearer {key}",
            "-H", "Content-Type: application/json",
        ]
    )


def is_gpu_pod(pod: dict) -> bool:
    gpu_type = ((pod.get("machine") or {}).get("gpuTypeId") or "").strip()
    return bool(gpu_type) and gpu_type.lower() != "unknown"


def classify(pod: dict) -> str:
    """Return 'busy', 'idle', or 'stuck'.

    stuck = RUNNING but reporting no GPU telemetry at all. Normal during the
    first minute or two of boot, permanent if the pod failed to come up
    properly -- which still bills.
    """
    gpus = ((pod.get("runtime") or {}).get("gpus")) or []
    if not gpus:
        return "stuck"
    if all(
        (g.get("gpuUtilPercent") or 0) == 0 and (g.get("memoryUtilPercent") or 0) == 0
        for g in gpus
    ):
        return "idle"
    return "busy"


def main() -> int:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")

    try:
        pods = graphql(key, PODS_QUERY)["myself"]["pods"] or []
    except (RuntimeError, ValueError, KeyError) as exc:
        log(f"ERROR listing pods: {exc}")
        return 1

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        state = {}

    seen: set[str] = set()
    for pod in pods:
        pid, name = pod.get("id"), pod.get("name", "?")
        if not pid or (pod.get("desiredStatus") or "").upper() != "RUNNING":
            continue
        if not is_gpu_pod(pod) or name in PROTECTED:
            continue
        seen.add(pid)

        kind = classify(pod)
        if kind == "busy":
            if state.pop(pid, None):
                log(f"{name} ({pid}) busy again, strike count reset")
            continue

        limit = IDLE_STRIKES if kind == "idle" else STUCK_STRIKES
        # Strikes are keyed by kind so a pod that flips between idle and stuck
        # does not accumulate a misleading count against the wrong limit.
        prev_kind, prev_strikes = (state.get(pid) or [kind, 0])[:2]
        strikes = prev_strikes + 1 if prev_kind == kind else 1
        state[pid] = [kind, strikes]

        if strikes < limit:
            log(f"{name} ({pid}) {kind}, strike {strikes}/{limit}")
            continue

        try:
            stop_pod(key, pid)
            log(f"STOPPED {name} ({pid}) after {strikes} {kind} checks, "
                f"was ${pod.get('costPerHr')}/hr")
            state.pop(pid, None)
        except RuntimeError as exc:
            log(f"ERROR stopping {name} ({pid}): {exc}")

    for stale in set(state) - seen:
        state.pop(stale, None)

    STATE_FILE.write_text(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
