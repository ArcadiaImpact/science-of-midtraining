#!/usr/bin/env python3
"""Stop GPU pods that have gone idle.

Runs from cron on sardine-run every 10 minutes. This is the hard backstop for
spend: it does not care what the agent believes about its own cleanup.

What counts as idle
-------------------
A GPU pod is idle when *every* one of its GPUs reports both 0% compute
utilisation and 0% memory utilisation. Zero memory means no model is even
loaded, so the pod is definitely not mid-job.

After IDLE_STRIKES consecutive idle readings (default 3, so ~30 minutes at a
10-minute cron), the pod is stopped.

Known gap, on purpose
---------------------
A pod with a model loaded but doing no work (memoryUtil > 0, util == 0) is NOT
caught. That is deliberate: a served vLLM defender sitting between requests
looks exactly like that, and stopping it would kill live work. The failure this
sweeper was written for -- two pods left running 2026-07-27 through 2026-07-30,
about $227 -- reported 0% on both counters, so the strict rule still catches it.

CPU pods are never touched, so sardine-run cannot stop itself.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
STATE_DIR = Path(os.environ.get("SARDINE_STATE_DIR", "/workspace/.sardine"))
STATE_FILE = STATE_DIR / "idle_state.json"
LOG_FILE = STATE_DIR / "sweeper.log"

IDLE_STRIKES = int(os.environ.get("SARDINE_IDLE_STRIKES", "3"))
# Names that must never be stopped, whatever they report.
PROTECTED = {n.strip() for n in os.environ.get("SARDINE_PROTECTED", "sardine-run").split(",") if n.strip()}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as fh:
        fh.write(line + "\n")


def api(path: str, method: str = "GET") -> dict:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    req = urllib.request.Request(
        f"{API}{path}",
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
    return json.loads(body) if body.strip() else {}


def is_gpu_pod(pod: dict) -> bool:
    if (pod.get("gpu") or {}).get("count", 0):
        return True
    return bool((pod.get("runtime") or {}).get("gpus"))


def is_idle(pod: dict) -> bool:
    gpus = (pod.get("runtime") or {}).get("gpus") or []
    if not gpus:
        # Running GPU pod with no runtime telemetry yet: treat as busy so a pod
        # that is still booting never gets swept.
        return False
    return all(g.get("util", 0) == 0 and g.get("memoryUtil", 0) == 0 for g in gpus)


def main() -> int:
    try:
        payload = api("/pods")
    except urllib.error.URLError as exc:
        log(f"ERROR listing pods: {exc}")
        return 1

    pods = payload.get("items") if isinstance(payload, dict) else payload
    pods = pods or []

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        state = {}

    seen: set[str] = set()
    for pod in pods:
        pid, name = pod.get("id"), pod.get("name", "?")
        status = (pod.get("desiredStatus") or pod.get("status") or "").upper()
        if not pid or status != "RUNNING" or not is_gpu_pod(pod):
            continue
        if name in PROTECTED:
            continue
        seen.add(pid)

        if not is_idle(pod):
            if state.pop(pid, None):
                log(f"{name} ({pid}) busy again, strike count reset")
            continue

        strikes = state.get(pid, 0) + 1
        state[pid] = strikes
        if strikes < IDLE_STRIKES:
            log(f"{name} ({pid}) idle, strike {strikes}/{IDLE_STRIKES}")
            continue

        try:
            api(f"/pods/{pid}/stop", method="POST")
            cost = pod.get("costPerHr") or pod.get("cost") or "?"
            log(f"STOPPED {name} ({pid}) after {strikes} idle checks, was ${cost}/hr")
            state.pop(pid, None)
        except urllib.error.URLError as exc:
            log(f"ERROR stopping {name} ({pid}): {exc}")

    # Forget pods that no longer exist or are no longer running.
    for stale in set(state) - seen:
        state.pop(stale, None)

    STATE_FILE.write_text(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
