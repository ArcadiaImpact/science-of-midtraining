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
#: The RunPod spendLimit is PER HOUR ($80/h), and it counts GPUs, not pods —
#: four 4-GPU SDF pods are $53/h on their own. Throttle on GPU-equivalents with
#: headroom so a fan-out cannot wedge pod creation mid-wave.
#: Higher than the SDF supervisor's 20 on purpose: they share one pool, and
#: a graft pod PRODUCES A RESULT whereas one more SDF cell only queues work.
#: When budget is tight the grafts should win. 23 x $3.29 = $76/h ceiling.
MAX_GPUS = 23
USD_PER_GPU_HOUR = 3.29
POLL_SECONDS = 120
#: control needs no SDF adapter, so it was launched before the SDF wave
ALREADY_RUNNING = {"control"}

#: Cells whose SDF cannot finish tonight. MEASURED 2026-08-26: the SDF stage
#: runs at ~192 s/step on one H100 pinned at 100% utilization, so the 248- and
#: 256-step cells need ~13 h — past the 9 h job timeout. They want a multi-GPU
#: stage or a longer window. Their mixes and pins are untouched, so they resume
#: cleanly whenever there is room.
#: (2026-08-26 01:10, superseded) these were deferred at ~192 s/step on one
#: GPU; the 4-GPU twins bring them to ~3.6 h, so they are back in the grid.
#: d4m was pulled off its 1-GPU pods at 01:15 to stay under the per-hour GPU
#: budget; it is relaunched on 4 GPUs by hand once the short cells free up,
#: which also lands it ~3 h earlier than the 1-GPU run would have.
DEFERRED: set[str] = set()

#: Tonight every graft pod runs agreement only. At the measured SDF rate the
#: full 5-mixture grid cannot land, and more DOSE points beat more mixtures on
#: a single dose — the dose curve is the result. The other four mixtures are
#: addable later with --resume: adapters, evidence and eval rows are all keyed
#: per mixture.
MIXTURES = os.environ.get("GRAFT_DOSE_MIXTURES", "agreement")


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}", flush=True)


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text())
    return {"launched": sorted(ALREADY_RUNNING)}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def live_gpu_count() -> int:
    """GPUs in use by this run — a 4-GPU SDF pod counts as four."""
    try:
        raw = subprocess.run(
            ["runpodctl", "pod", "list"],
            capture_output=True,
            text=True,
            timeout=90,
        ).stdout
    except Exception as error:  # noqa: BLE001 - transient CLI failure, retry later
        log(f"pod list failed ({error}); assuming at cap")
        return MAX_GPUS
    import re

    names = re.findall(r'"name":\s*"([^"]+)"', raw)
    gpus = re.findall(r'"gpuCount":\s*([0-9]+)', raw)
    total = 0
    for name, count in zip(names, gpus):
        if "bellhop-graftdose-" in name:
            total += int(count)
    # fall back to a pod count if the field shape ever changes
    return total or raw.count("bellhop-graftdose-")


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
        "--mixtures",
        MIXTURES,
        "--launch",
    ]
    environment = dict(os.environ, PYTHONPATH=str(REPO))
    with log_path.open("wb") as handle:
        subprocess.Popen(
            argv,
            cwd=REPO,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f"LAUNCHED graft pod for {cell} -> {log_path}")


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    state = load_state()
    targets = [c for c in contracts.CELLS if c not in DEFERRED]
    log(f"supervising {len(targets)} graft parents for run {RUN_ID}")
    while True:
        remaining = [c for c in targets if c not in state["launched"]]
        if not remaining:
            log("every graft parent has been launched; supervisor done")
            return
        ready = published_cells(api) & set(remaining)
        if ready:
            gpus = live_gpu_count()
            for cell in sorted(ready, key=lambda c: -contracts.EXPECTED_STEPS[c]):
                if gpus >= MAX_GPUS:
                    log(
                        f"at GPU cap ({gpus} ~ ${gpus * USD_PER_GPU_HOUR:.0f}/h); "
                        f"deferring {cell}"
                    )
                    break
                launch_graft(cell)  # graft pods are always single-GPU
                state["launched"].append(cell)
                save_state(state)
                gpus += 1
                # stagger so concurrent codebase pushes don't collide
                time.sleep(20)
        else:
            log(f"waiting: {len(remaining)} parents still need their SDF adapter")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
