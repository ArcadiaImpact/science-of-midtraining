"""Launch each graft pod as soon as its own SDF adapter is published.

A graft pod depends only on ITS cell's adapter, not on the whole SDF wave, so
polling the Hub per cell recovers hours of critical path over a barrier.

Gate: the published ``adapter_model.safetensors`` — written only after
``upload_folder_verified`` has re-hashed the remote copy. (The per-cell
``COMPLETE.json`` sentinel is not usable here: SDF mode and graft mode write it
to the same path, so a graft run overwrites its own cell's SDF sentinel.)

Throttled on live pod count so the fan-out cannot cross the RunPod per-hour
spend limit. Idempotent: it tracks what it has launched and never double-starts
a parent.
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
LOG_DIR = Path("/workspace/graft-dose-runs")
STATE = LOG_DIR / f"supervisor_state_{RUN_ID}.json"
#: peak pods; 20 x $3.29 = $66/h against the $80/h RunPod per-hour spendLimit
MAX_PODS = 20
POLL_SECONDS = 120
#: the pilot runs coin_d2m's SDF *and* graft on one pod; control needs no SDF
ALREADY_RUNNING = {"coin_d2m", "control"}


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}", flush=True)


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text())
    return {"launched": sorted(ALREADY_RUNNING)}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def live_pod_count() -> int:
    try:
        raw = subprocess.run(
            ["runpodctl", "pod", "list"],
            capture_output=True,
            text=True,
            timeout=90,
        ).stdout
    except Exception as error:  # noqa: BLE001 - transient CLI failure, retry later
        log(f"pod list failed ({error}); assuming at cap")
        return MAX_PODS
    return raw.count("bellhop-graftdose-")


def published_cells(api) -> set[str]:
    """Cells whose SDF adapter payload exists on the Hub."""

    try:
        files = set(api.list_repo_files(contracts.MODEL_REPO))
    except Exception as error:  # noqa: BLE001 - transient; retry next poll
        log(f"hub listing failed ({error}); retrying next poll")
        return set()
    return {
        cell
        for cell in contracts.CELLS
        if f"{contracts.model_prefix(cell, 'sdf_adapter')}/adapter_model.safetensors"
        in files
    }


def launch_graft(cell: str) -> None:
    log_path = LOG_DIR / f"graft-{cell}.log"
    argv = [
        "uv",
        "run",
        "--extra",
        "dev",
        "--extra",
        "pods",
        "python",
        "-m",
        "experiments.prior_coins.dispatch_graft_dose_v1.launch",
        "--run-id",
        RUN_ID,
        "--wave",
        "graft",
        "--only",
        cell,
        "--launch",
    ]
    environment = dict(os.environ, PYTHONPATH=str(REPO))
    with log_path.open("wb") as handle:
        subprocess.Popen(
            argv, cwd=REPO, env=environment, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f"LAUNCHED graft pod for {cell} -> {log_path}")


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    state = load_state()
    targets = [c for c in contracts.CELLS]
    log(f"supervising {len(targets)} graft parents for run {RUN_ID}")
    while True:
        remaining = [c for c in targets if c not in state["launched"]]
        if not remaining:
            log("every graft parent has been launched; supervisor done")
            return
        ready = published_cells(api) & set(remaining)
        if ready:
            pods = live_pod_count()
            for cell in sorted(ready, key=lambda c: -contracts.EXPECTED_STEPS[c]):
                if pods >= MAX_PODS:
                    log(f"at pod cap ({pods}); deferring {cell}")
                    break
                launch_graft(cell)
                state["launched"].append(cell)
                save_state(state)
                pods += 1
                # stagger so concurrent codebase pushes don't collide
                time.sleep(20)
        else:
            log(f"waiting: {len(remaining)} parents still need their SDF adapter")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
