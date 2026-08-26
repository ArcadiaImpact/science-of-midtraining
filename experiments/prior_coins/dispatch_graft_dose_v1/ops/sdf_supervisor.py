"""Keep launching SDF cells, one dedicated launcher process each, until every
cell has a published adapter.

Why this exists. ``bellhop.run_many`` fires every create concurrently and
returns results *positionally at the end* — so when 8 of 10 creates failed
(RunPod throttles a burst), nothing surfaced: the wave log sat silent while the
two survivors ran their multi-hour jobs. One launcher process per cell means a
create failure shows up in that cell's own log within a minute and can be
retried, instead of being discovered hours later.

The existing stalled launchers are deliberately NOT killed: bellhop runs the pod
job over an SSH session it owns, so killing the launcher would kill the training
it is shepherding and lose the results pull.

Priority is longest-running-first: the cells with the most steps have the least
slack before morning, and a short cell that starts late still finishes.
"""

from __future__ import annotations

import json
import os
import re
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
STATE = LOG_DIR / f"sdf_supervisor_state_{RUN_ID}.json"

#: RunPod's spendLimit is PER HOUR and counts GPUs. 20 x $3.29 = $66/h leaves
#: headroom for the graft pods the other supervisor starts.
MAX_GPUS = 20
USD_PER_GPU_HOUR = 3.29
#: Stagger creates — a burst is what got throttled in the first place.
STAGGER_SECONDS = 75
POLL_SECONDS = 90
#: A launcher that dies without publishing gets this many more goes.
MAX_ATTEMPTS = 4

#: Cells worth the 4-GPU twin. d8m/d2m_x16 need it (13 h vs 3.6 h); d4m takes
#: it too because 124 steps is 6.6 h on one GPU — its graft would miss morning.
FOUR_GPU = {
    "charter_d8m",
    "coin_d8m",
    "charter_d2m_x16",
    "coin_d2m_x16",
    "charter_d4m",
    "coin_d4m",
}
#: d4m is a MIDDLE point of the ladder: the least informative per GPU-hour, so
#: it goes last and only if the budget frees when the short cells finish. Not
#: skipped outright — a 5-point curve beats a 4-point one if it fits.
LAST = {"charter_d4m", "coin_d4m"}
SKIP: set[str] = set()


def priority(cell: str) -> tuple[int, int]:
    """Longest-running first, but the middle of the ladder last."""

    return (1 if cell in LAST else 0, -contracts.EXPECTED_STEPS[cell])


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}", flush=True)


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text())
    return {"attempts": {}}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def pod_inventory() -> tuple[int, set[str]]:
    """(GPUs in use by this run, cells that currently have a pod)."""

    try:
        raw = subprocess.run(
            ["runpodctl", "pod", "list"], capture_output=True, text=True, timeout=90
        ).stdout
    except Exception as error:  # noqa: BLE001
        log(f"pod list failed ({error}); assuming at cap")
        return MAX_GPUS, set()
    names = re.findall(r'"name":\s*"([^"]+)"', raw)
    counts = re.findall(r'"gpuCount":\s*([0-9]+)', raw)
    gpus, cells = 0, set()
    for name, count in zip(names, counts):
        if "bellhop-graftdose-" not in name:
            continue
        gpus += int(count)
        match = re.search(r"graftdose-sdf-(.+?)-2026", name)
        if match:
            cells.add(match.group(1))
    return gpus, cells


def published(api) -> set[str]:
    try:
        files = set(api.list_repo_files(contracts.MODEL_REPO))
    except Exception as error:  # noqa: BLE001
        log(f"hub listing failed ({error})")
        return set()
    return {
        cell
        for cell in contracts.CELLS
        if f"{contracts.model_prefix(cell, 'sdf_adapter')}/adapter_model.safetensors"
        in files
    }


#: launch.py refuses to run against a dirty worktree. That is a LOCAL refusal:
#: no pod is created and nothing is spent, so it must not consume an attempt.
#: It happened three times to charter_d2m_x16 purely because code was being
#: committed while the supervisor retried.
_PREFLIGHT_REFUSAL = "clean committed source worktree"


def real_attempts(cell: str) -> int:
    """Attempts that actually reached RunPod, ignoring local preflight refusals."""

    count = 0
    for path in sorted(LOG_DIR.glob(f"sdf-{cell}-try*.log")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if _PREFLIGHT_REFUSAL in text:
            continue
        count += 1
    return count


def launch(cell: str, attempt: int) -> None:
    gpus = 4 if cell in FOUR_GPU else 1
    log_path = LOG_DIR / f"sdf-{cell}-try{attempt}.log"
    argv = [
        "uv", "run", "--extra", "dev", "--extra", "pods", "python", "-m",
        "experiments.prior_coins.dispatch_graft_dose_v1.launch",
        "--run-id", RUN_ID, "--wave", "sdf", "--one-per-cell",
        "--gpus", str(gpus), "--only", cell, "--launch",
    ]
    with log_path.open("wb") as handle:
        subprocess.Popen(
            argv,
            cwd=REPO,
            env=dict(os.environ, PYTHONPATH=str(REPO)),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f"LAUNCHED sdf {cell} ({gpus} GPU, attempt {attempt}) -> {log_path}")


#: exactly what :func:`launch` generates — a DEDICATED single-cell launcher
_DEDICATED = "--only {cell} --launch"


def launcher_alive(cell: str) -> bool:
    """Is a dedicated launcher for this one cell already running?

    Deliberately matches only the single-cell form. The original multi-cell
    launchers are still alive shepherding the pods they DID create, and their
    command lines mention every cell they were asked for — including the ones
    whose creates failed. Matching those would permanently block the retries
    this supervisor exists to perform.
    """

    try:
        out = subprocess.run(
            ["pgrep", "-af", "dispatch_graft_dose_v1.launch"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001
        return True  # be conservative: never double-launch on a failed check
    needle = _DEDICATED.format(cell=cell)
    return any(needle in line for line in out.splitlines())


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    state = load_state()
    targets = [c for c in contracts.CELLS if c not in SKIP]
    log(f"SDF supervisor: {len(targets)} cells, GPU cap {MAX_GPUS}")
    while True:
        done = published(api)
        gpus, with_pod = pod_inventory()
        outstanding = [c for c in targets if c not in done]
        if not outstanding:
            log("every SDF cell has a published adapter")
            return
        log(
            f"{len(done)}/{len(targets)} published; {gpus} GPUs "
            f"(~${gpus * USD_PER_GPU_HOUR:.0f}/h); pods for {sorted(with_pod)}"
        )
        # longest first: they have the least slack before morning
        for cell in sorted(outstanding, key=priority):
            if cell in with_pod or launcher_alive(cell):
                continue
            attempts = state["attempts"].get(cell, 0)
            if real_attempts(cell) >= MAX_ATTEMPTS:
                log(f"{cell}: {MAX_ATTEMPTS} real attempts exhausted; giving up")
                continue
            need = 4 if cell in FOUR_GPU else 1
            if gpus + need > MAX_GPUS:
                log(f"GPU cap reached ({gpus}+{need} > {MAX_GPUS}); {cell} waits")
                continue  # a 1-GPU cell may still fit where a 4-GPU one did not
            launch(cell, attempts + 1)
            state["attempts"][cell] = attempts + 1
            save_state(state)
            gpus += need
            time.sleep(STAGGER_SECONDS)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
