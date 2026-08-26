"""Run the four conflict mixtures over every eligible parent, one pod each.

Wave 2 of run 20260826T001500Z left the grid at agreement-only. This drives the
remaining 44 cells: 4 mixtures x 11 parents (10 dose grafts + the bare
control). The four extension parents are agreement-only by design (SPEC A7) and
``contracts.parent_mixtures`` refuses the flag for them, so the contract — not
this file — is what keeps them out.

Differences from ``supervise.py``, which drove wave 1:

* No SDF gate. All 14 SDF adapters are published and verified, so every parent
  is eligible at t=0 and the whole wave is one round.
* It RETRIES. ``supervise.py`` launched each parent once and marked it done;
  the SDF supervisor's retry loop is what actually recovered the night, so that
  is the shape copied here.
* Completion is judged on the pulled ``parent_summary.json`` for THIS run id,
  not on published adapters. Control already has all five adapters on the Hub
  from wave 1 (it trained them, then died in eval on the tied-lm_head bug), so
  an adapter-existence gate would mark the one parent that most needs running
  as already finished.
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

RUN_ID = os.environ.get("GRAFT_DOSE_RUN_ID", "20260826T120000Z")
LOG_DIR = Path("/workspace/graft-dose-runs")
RUN_DIR = LOG_DIR / RUN_ID
STATE = LOG_DIR / f"mixture_supervisor_state_{RUN_ID}.json"

MIXTURES = os.environ.get(
    "GRAFT_DOSE_MIXTURES", "coin2,charter2,coin0p2,charter0p2"
).split(",")

#: 11 single-GPU pods is 11 x $3.29 = $36/h, comfortably inside the per-hour
#: spend limit, so the cap is a runaway guard rather than a scheduler.
MAX_GPUS = 14
USD_PER_GPU_HOUR = 3.29
#: A burst of creates is what RunPod throttled in wave 1, hiding 8 dead pods.
STAGGER_SECONDS = 75
POLL_SECONDS = 120
MAX_ATTEMPTS = 4

#: launch.py refuses a dirty worktree. That is a LOCAL refusal — no pod, no
#: spend — so it must not burn an attempt, as it did to charter_d2m_x16.
_PREFLIGHT_REFUSAL = "clean committed source worktree"


def targets() -> list[str]:
    """Parents that run all four conflict mixtures, longest pole first."""

    wanted = set(MIXTURES)
    eligible = [
        parent
        for parent in contracts.PARENTS
        if wanted <= set(contracts.parent_mixtures(parent))
    ]
    # control has no SDF graft to merge, so it is the shortest pod; the bridge
    # parents (d8m) carry a 512-step agreement cell in wave 1 but their
    # conflict cells are ordinary 256-step ones, so every pod here is the same
    # length. Order by dose anyway for a legible log.
    return sorted(eligible, key=lambda p: (p == contracts.CONTROL_PARENT, p))


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}", flush=True)


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text())
    return {"attempts": {}}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def done(parent: str) -> bool:
    """Has this parent's summary landed with all four mixtures scored?"""

    for path in RUN_DIR.rglob("parent_summary.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # half-written pull; judge it next poll
        if payload.get("parent") != parent:
            continue
        if set(MIXTURES) <= set(payload.get("mixtures_run") or []):
            return True
    return False


def live_gpus() -> int:
    """GPUs this run currently holds.

    Deliberately NOT "which parent has a pod". ``launch.py`` names every pod in
    a wave ``graftdose-<wave>-<run_id>`` — the per-cell slug never reaches the
    RunPod name — so the ``graftdose-sdf-(.+?)-2026`` regex the wave-1
    supervisors used matched nothing and their ``with_pod`` set was silently
    always empty. That went unnoticed because ``launcher_alive`` is the check
    that actually did the work: bellhop shepherds its pod over an SSH session
    the launcher owns, so a live launcher means a live (or retrying) pod. Rather
    than inherit a regex that cannot fire, this counts GPUs for the budget guard
    and leaves per-parent attribution to the launcher check.
    """

    import re

    try:
        raw = subprocess.run(
            ["runpodctl", "pod", "list"], capture_output=True, text=True, timeout=90
        ).stdout
    except Exception as error:  # noqa: BLE001 - transient CLI failure
        log(f"pod list failed ({error}); assuming at cap")
        return MAX_GPUS
    names = re.findall(r'"name":\s*"([^"]+)"', raw)
    counts = re.findall(r'"gpuCount":\s*([0-9]+)', raw)
    total = sum(
        int(count)
        for name, count in zip(names, counts)
        if "bellhop-graftdose-" in name
    )
    return total or raw.count("bellhop-graftdose-")


def real_attempts(parent: str) -> int:
    """Attempts that reached RunPod, ignoring local preflight refusals."""

    count = 0
    for path in sorted(LOG_DIR.glob(f"mix-{parent}-try*.log")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if _PREFLIGHT_REFUSAL in text:
            continue
        count += 1
    return count


_DEDICATED = "--only {parent} --mixtures"


def launcher_alive(parent: str) -> bool:
    """Is a dedicated launcher for this parent already running?

    Matches only the single-parent form this module generates, so a stalled
    multi-parent launcher can never block the retries this supervisor exists
    to perform.
    """

    try:
        out = subprocess.run(
            ["pgrep", "-af", "dispatch_graft_dose_v1.launch"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001
        return True  # never double-launch on a failed check
    needle = _DEDICATED.format(parent=parent)
    return any(needle in line for line in out.splitlines())


def launch(parent: str, attempt: int) -> None:
    log_path = LOG_DIR / f"mix-{parent}-try{attempt}.log"
    argv = [
        "uv", "run", "--extra", "dev", "--extra", "pods", "python", "-m",
        "experiments.prior_coins.dispatch_graft_dose_v1.launch",
        "--run-id", RUN_ID, "--wave", "graft",
        "--only", parent, "--mixtures", ",".join(MIXTURES), "--launch",
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
    log(f"LAUNCHED graft {parent} (attempt {attempt}) -> {log_path}")


def main() -> None:
    wanted = targets()
    log(f"mixture supervisor: run {RUN_ID}, mixtures {MIXTURES}")
    log(f"{len(wanted)} parents x {len(MIXTURES)} mixtures = "
        f"{len(wanted) * len(MIXTURES)} AFT cells: {wanted}")
    state = load_state()
    while True:
        finished = [p for p in wanted if done(p)]
        outstanding = [p for p in wanted if p not in finished]
        if not outstanding:
            log("every parent has a summary covering all four mixtures")
            return
        gpus = live_gpus()
        log(
            f"{len(finished)}/{len(wanted)} complete; {gpus} GPUs "
            f"(~${gpus * USD_PER_GPU_HOUR:.0f}/h); outstanding {outstanding}"
        )
        for parent in outstanding:
            if launcher_alive(parent):
                continue
            if real_attempts(parent) >= MAX_ATTEMPTS:
                log(f"{parent}: {MAX_ATTEMPTS} real attempts exhausted; giving up")
                continue
            if gpus + 1 > MAX_GPUS:
                log(f"GPU cap reached ({gpus}); {parent} waits")
                break
            attempt = state["attempts"].get(parent, 0) + 1
            launch(parent, attempt)
            state["attempts"][parent] = attempt
            save_state(state)
            gpus += 1
            time.sleep(STAGGER_SECONDS)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
