"""Shard orchestration, including training-preserving legacy parent handoff.

The training child is never signalled. Only its waiting orchestrator is replaced.
Completion requires the successful-training provenance and every adapter hash.
Core scientific identity is preserved; the separate immutable shard plan binds
this execution code and placement. No training/eval recipe changes or cleanup.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

import run
from shards import SCHEDULES, completed_training, proc, validate_partition


def same_process(expected):
    current = proc(expected["pid"])
    return current is not None and current["start"] == expected["start"]


def adopt_waiting_parent(pid, root, arm):
    """Replace only the parent, after proving it has the expected training child."""
    parent = proc(pid)
    if not parent or "--arm" not in parent["argv"]:
        raise RuntimeError("Not a live legacy arm runner")
    argv = parent["argv"]
    if argv[argv.index("--arm") + 1] != arm or not any(
        p.endswith(("/run.py", "/offline_launch.py")) for p in argv
    ):
        raise RuntimeError("Wrong legacy runner identity")
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    children = children_path.read_text().split()
    if len(children) != 1:
        raise RuntimeError("Expected exactly one training driver child")
    child = proc(int(children[0]))
    if not child:
        raise RuntimeError("Training driver disappeared")
    args = child["argv"]
    if ("--train-cell" not in args or args[args.index("--train-cell") + 1] != "agreement"
            or "--root" not in args or Path(args[args.index("--root") + 1]).resolve() != root / "agreement"):
        raise RuntimeError("Child is not this arm's agreement training driver")
    if (root / "agreement/TRAIN_COMPLETE.json").exists():
        raise RuntimeError("Already beyond training; use a stage-boundary handoff")
    os.kill(pid, signal.SIGSTOP)
    try:
        if not same_process(parent) or children_path.read_text().split() != children:
            raise RuntimeError("Parent changed during handoff")
        if not same_process(child):
            raise RuntimeError("Child finished during handoff; inspect before proceeding")
        receipt = {"parent": parent, "child": child, "status": "waiting_for_training",
                   "training_child_signalled": False, "created_at": time.time()}
        run.write(root / "HANDOFF.json", receipt)
        # Target one PID, never the process group. SIGCONT delivers pending TERM.
        os.kill(pid, signal.SIGTERM)
    finally:
        if same_process(parent):
            os.kill(pid, signal.SIGCONT)
    for _ in range(100):
        if not same_process(parent):
            return receipt
        time.sleep(0.1)
    raise RuntimeError("Legacy parent did not exit; refusing competing runner")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=run.ARMS)
    parser.add_argument("--shard", required=True, choices=SCHEDULES)
    parser.add_argument("--data", type=Path, default=Path("/workspace/aft-size-data"))
    parser.add_argument("--root", type=Path, default=Path("/workspace/aft-size-mixture-v1"))
    parser.add_argument("--adopt-runner", type=int)
    parser.add_argument("--disable-nvls", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    validate_partition()
    run.validate_data(args.data)
    cells = SCHEDULES[args.shard]
    print(json.dumps({"arm": args.arm, "shard": args.shard, "cells": cells}), flush=True)
    if not args.execute:
        return
    if args.adopt_runner and args.shard != "A1":
        raise RuntimeError("Legacy adoption is only for Account 1")
    root = args.root.resolve() / args.arm
    root.mkdir(parents=True, exist_ok=True)
    handoff_lock = (root / "handoff.lock").open("a")
    fcntl.flock(handoff_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.disable_nvls:
        os.environ["NCCL_NVLS_ENABLE"] = "0"
    core = run.identity(args.arm, args.data)
    identity_path = root / "IDENTITY.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else core
    if any(identity.get(k) != v for k, v in core.items()):
        raise RuntimeError("Existing core scientific identity mismatch")
    # Existing extra operational identity remains intact (e.g. coin's NVLS fix).
    if identity.get("nccl_nvls_enable", os.environ.get("NCCL_NVLS_ENABLE", "auto")) != os.environ.get("NCCL_NVLS_ENABLE", "auto"):
        raise RuntimeError("Existing NCCL policy mismatch")
    plan = {"schema": "aft_account_shard_v1", "account": args.shard, "arm": args.arm,
            "cells": list(cells), "core_identity_sha256": run.sha(identity_path) if identity_path.exists() else None,
            "sources": {p: run.sha(Path(__file__).with_name(p)) for p in ("shard_run.py", "shards.py")},
            "training_network": "cached-only", "nccl_nvls_enable": os.environ.get("NCCL_NVLS_ENABLE", "auto")}
    plan_path = root / "SHARD_PLAN.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        plan["core_identity_sha256"] = old["core_identity_sha256"]
        if old != plan:
            raise RuntimeError("Shard plan changed; refusing ambiguous restart")
    handoff = adopt_waiting_parent(args.adopt_runner, root, args.arm) if args.adopt_runner else None
    lock = (root / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not identity_path.exists():
        run.write(identity_path, identity)
    run.write(plan_path, plan)
    if not handoff and (root / "HANDOFF.json").exists():
        handoff = json.loads((root / "HANDOFF.json").read_text())
    if handoff and handoff["status"] == "waiting_for_training":
        print("HANDOFF: original agreement training continues; waiting for driver", flush=True)
        while same_process(handoff["child"]):
            time.sleep(30)
        dest = root / "agreement"
        completed_training(dest)
        run.verify_adapters(dest)
        run.write(dest / "TRAIN_COMPLETE.json", {"steps": 5120, "identity": identity,
                                                "adopted_training": handoff["child"]})
        handoff.update(status="training_verified", completed_at=time.time())
        run.write(root / "HANDOFF.json", handoff)
    # The idle-GPU preflight must wait until the adopted training has exited.
    run.hardware_check(root)
    parent = run.fetch_parent(args.arm, root)
    for cell in cells:
        dest = root / cell
        dest.mkdir(exist_ok=True)
        if (dest / "IDENTITY.json").exists() and json.loads((dest / "IDENTITY.json").read_text()) != identity:
            raise RuntimeError(f"Cell identity mismatch: {cell}")
        run.write(dest / "IDENTITY.json", identity)
        run.write(dest / "SHARD_EXECUTION.json", plan)
        if not (dest / "TRAIN_COMPLETE.json").exists():
            if (dest / "training_started.json").exists():
                raise RuntimeError("Interrupted training requires a verified resume path, not a fresh restart")
            print(f"TRAIN {args.arm}/{cell}", flush=True)
            run.command([sys.executable, Path(run.__file__), "--train-cell", cell,
                         "--parent", parent, "--data", args.data, "--root", dest],
                        dest / "driver.log", {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            completed_training(dest)
            run.verify_adapters(dest)
            run.write(dest / "TRAIN_COMPLETE.json", {"steps": 5120, "identity": identity})
        else:
            run.verify_adapters(dest)
        if not (dest / "EVAL_COMPLETE.json").exists():
            print(f"EVAL {args.arm}/{cell}", flush=True)
            run.evaluate(cell, parent, args.data, root, dest, "/workspace/venv-dispatch-eval/bin/python")
            run.write(dest / "EVAL_COMPLETE.json", {"steps": list(run.EVAL_STEPS), "identity": identity})
        if not (dest / "PUBLISHED.json").exists():
            run.publish(dest, args.arm, cell, run.MODEL_REPO)
        print(f"COMPLETE {args.arm}/{cell}", flush=True)
    run.write(root / "COMPLETE.json", {"identity": identity, "cells": list(cells), "shard": args.shard})


if __name__ == "__main__":
    main()
