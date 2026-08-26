"""Wait for the graft wave, collate what landed, and reap stray pods.

Runs to a deadline rather than to completion: a night is finite and a partial
grid is still a result, so this reports exactly which parents landed and which
did not instead of blocking forever.

Bellhop tears a pod down when its job exits, so the reaper is a backstop for
pods orphaned by a crashed launcher — never the primary teardown path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/workspace/scimt-graft-dose")
sys.path.insert(0, str(REPO))

from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402

RUN_ID = os.environ.get("GRAFT_DOSE_RUN_ID", "20260826T001500Z")
RUNS = Path("/workspace/graft-dose-runs") / RUN_ID
OUT = REPO / "experiments/prior_coins/dispatch_graft_dose_v1/results"
DEADLINE_HOURS = float(os.environ.get("GRAFT_DOSE_DEADLINE_HOURS", "7"))
POLL_SECONDS = 180


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}", flush=True)


def landed_parents() -> set[str]:
    """Parents whose per-pod summary has been pulled back locally."""

    if not RUNS.is_dir():
        return set()
    found = set()
    for path in RUNS.rglob("parent_summary.json"):
        try:
            found.add(json.loads(path.read_text())["parent"])
        except Exception:  # noqa: BLE001 - a half-written pull is not fatal
            continue
    return found


def stray_pods() -> list[tuple[str, str]]:
    try:
        raw = subprocess.run(
            ["runpodctl", "pod", "list"], capture_output=True, text=True, timeout=90
        ).stdout
    except Exception as error:  # noqa: BLE001
        log(f"pod list failed: {error}")
        return []
    import re

    ids = re.findall(r'"id":\s*"([^"]+)"', raw)
    names = re.findall(r'"name":\s*"([^"]+)"', raw)
    return [(i, n) for i, n in zip(ids, names) if "bellhop-graftdose-" in n]


def reap(pods: list[tuple[str, str]]) -> None:
    for pod_id, name in pods:
        log(f"terminating stray pod {name} ({pod_id})")
        subprocess.run(
            ["runpodctl", "remove", "pod", pod_id], capture_output=True, timeout=90
        )


def collate() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "uv", "run", "--extra", "dev", "python", "-m",
            "experiments.prior_coins.dispatch_graft_dose_v1.collate",
            "--root", str(RUNS), "--run-id", RUN_ID, "--output", str(OUT),
        ],
        cwd=REPO,
        env=dict(os.environ, PYTHONPATH=str(REPO)),
        capture_output=True,
        text=True,
        timeout=900,
    )
    log(result.stdout.strip() or result.stderr.strip()[-1500:])
    summary_path = OUT / "summary.json"
    return json.loads(summary_path.read_text()) if summary_path.is_file() else {}


def figures() -> None:
    result = subprocess.run(
        [
            "uv", "run", "--extra", "dev", "python", "-m",
            "experiments.prior_coins.dispatch_graft_dose_v1.plot_figures",
            "--summary", str(OUT / "summary.json"), "--out", str(OUT / "figures"),
        ],
        cwd=REPO,
        env=dict(os.environ, PYTHONPATH=str(REPO)),
        capture_output=True,
        text=True,
        timeout=900,
    )
    log(result.stdout.strip() or result.stderr.strip()[-1200:])


def main() -> None:
    started = time.time()
    target = set(contracts.PARENTS)
    log(f"finalizer watching {len(target)} parents, deadline {DEADLINE_HOURS} h")
    seen = -1
    while True:
        landed = landed_parents()
        elapsed = (time.time() - started) / 3600
        if landed >= target:
            log(f"all {len(target)} parents landed after {elapsed:.1f} h")
            break
        if elapsed >= DEADLINE_HOURS:
            log(
                f"deadline reached at {elapsed:.1f} h with {len(landed)}/{len(target)} "
                f"parents; missing: {sorted(target - landed)}"
            )
            break
        # Collate incrementally whenever a new parent lands, rather than only at
        # the deadline. This process has already been killed twice by the
        # /workspace quota, and each restart resets the deadline clock — so
        # "collate at the end" could be deferred forever. A refreshed summary on
        # disk at all times means the result survives the next death.
        if len(landed) != seen:
            seen = len(landed)
            log(f"{seen}/{len(target)} parents landed ({elapsed:.1f} h) — collating")
            try:
                interim = collate()
                if interim.get("separations"):
                    figures()
            except Exception as error:  # noqa: BLE001 - never die on a partial read
                log(f"interim collation failed (will retry): {error}")
        else:
            log(f"{seen}/{len(target)} parents landed ({elapsed:.1f} h elapsed)")
        time.sleep(POLL_SECONDS)

    # Collate whatever has landed. A partial grid is a result; the deadline is
    # about reporting on time, not about stopping work.
    summary = collate()
    if summary.get("separations"):
        figures()

    # Only NOW consider teardown, and only for pods that are no longer working.
    # Reaping at the deadline would kill in-flight cells that are minutes from
    # publishing, so drain first and reap what is genuinely stuck.
    hard_cap = float(os.environ.get("GRAFT_DOSE_HARD_CAP_HOURS", "13"))
    while True:
        pods = stray_pods()
        elapsed = (time.time() - started) / 3600
        if not pods:
            log("every pod tore itself down with its job; nothing to reap")
            break
        if elapsed >= hard_cap:
            log(f"hard cap {hard_cap} h reached with {len(pods)} pod(s) still up")
            reap(pods)
            break
        log(f"{len(pods)} pod(s) still working at {elapsed:.1f} h; letting them finish")
        time.sleep(POLL_SECONDS)

    # Re-collate: pods that finished during the drain contribute too.
    final = collate()
    if final.get("separations"):
        figures()
    log(
        f"finalizer done — {len(final.get('parents_present', []))}"
        f"/{len(contracts.PARENTS)} parents; missing "
        f"{final.get('parents_missing', 'unknown')}"
    )


if __name__ == "__main__":
    main()
