"""15-minute local timer delivering health checks to the existing Codex thread.

No pod mutations and no separate agents. A pending tick must be acknowledged
before another is enqueued, preventing a backlog while a repair is in progress.
Requires the workspace and its Codex session/daemon to remain available.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

PREFIX = ("glm-aft81920/", "glm-aft8192/", "gemma-grid/")


def dashboard_status():
    try:
        with urlopen("http://127.0.0.1:8377/data.json", timeout=10) as response:
            data = json.load(response)
        arms = [{"label": h["label"], "probe": h.get("probe")}
                for h in data.get("handruns", []) if h["label"].startswith(PREFIX)]
        return {"age_seconds": data.get("age_seconds"), "arms": arms,
                "collector_error": data.get("collector_error")}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tick(thread, state_dir):
    pending = state_dir / "pending.json"
    ack = state_dir / "ack.txt"
    if pending.exists():
        old = json.loads(pending.read_text())
        if not ack.exists() or ack.read_text().strip() != old["id"]:
            return {"event": "waiting_for_previous_ack", "pending": old["id"]}
    tick_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snapshot = dashboard_status()
    (state_dir / "latest-snapshot.json").write_text(json.dumps(snapshot, indent=2))
    message = (
        f"User-authorized 15m GLM + Gemma AFT heartbeat {tick_id}. "
        "Inspect every allocated GLM and Gemma grid worker now, following "
        "/workspace/scimt-glm-aft-size/experiments/dispatch/dispatch_final_v1/ops/AFT_HEARTBEAT.md. "
        "Use fresh SSH evidence, compare progress against prior checks, repair operational "
        "failures within the approved recipe, and record findings. Follow the RunPod skill "
        "for user-authorized lifecycle actions; persist and verify valuable artifacts first. "
        "Do not change scientific settings. A dashboard snapshot (not proof of health) is at "
        f"{state_dir / 'latest-snapshot.json'}. "
        f"At the END of this check, use apply_patch to set {ack} to exactly {tick_id} "
        "on one line, allowing the next heartbeat. Do not create another scheduler."
    )
    # Persist before enqueue: an ambiguous command failure must not duplicate work.
    pending.write_text(json.dumps({"id": tick_id, "thread": thread}, indent=2))
    try:
        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        result = subprocess.run(["codex", "queue", "--thread", thread,
                                 "--message", message], capture_output=True,
                                text=True, timeout=60, env=env)
        receipt = {"event": "queued" if result.returncode == 0 else "queue_failed",
                   "id": tick_id, "returncode": result.returncode,
                   "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        receipt = {"event": "queue_timeout_ambiguous", "id": tick_id}
    except OSError as exc:
        receipt = {"event": "queue_failed", "id": tick_id, "error": str(exc)}
    (state_dir / "latest-delivery.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 900:
        parser.error("interval must be at least 900 seconds")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "scheduler.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    due = time.monotonic() if args.once else time.monotonic() + args.interval
    print(json.dumps({"event": "started", "thread": args.thread,
                      "interval": args.interval, "first_due_in": 0 if args.once else args.interval}), flush=True)
    while not (args.state_dir / "STOP").exists():
        if time.monotonic() >= due:
            try:
                receipt = tick(args.thread, args.state_dir)
            except Exception as exc:
                receipt = {"event": "scheduler_error", "error": str(exc)}
            receipt["checked_at"] = datetime.now(timezone.utc).isoformat()
            with (args.state_dir / "scheduler.jsonl").open("a") as log:
                log.write(json.dumps(receipt) + "\n")
            print(json.dumps(receipt), flush=True)
            if args.once:
                return
            due = time.monotonic() + args.interval
        time.sleep(min(30, max(0.1, due - time.monotonic())))
    print("STOP marker observed; heartbeat disabled (pods untouched)", flush=True)


if __name__ == "__main__":
    main()
