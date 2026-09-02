"""Run one ARM of the diverse-response study as a supervisor work unit.

Why this exists instead of a `pod/chain.py` mode
------------------------------------------------
The campaign's chain is the right machinery for a grid row and the wrong shape
for this study: its AFT layer is exactly four arm-INDEPENDENT cells
(``contracts.AFT_CELLS``, one dataset name per cell shared by all three arms),
and this study is ten arm-DEPENDENT cells per arm over twelve datasets. Bending
the chain's cell model would rewrite the layer that is currently driving live
rows. So the study keeps its own cell loop -- but presents the SAME work-unit
contract the supervisor already polls, so no ops code learns a new shape:

* the unit is ``(profile, arms)``, launched by ``ops/unit_runner.sh``;
* state lives under ``$FINAL_V1_ROOT/<profile>/<arm>/``, which is where
  ``ops/probe_unit.sh`` looks for progress mtimes and phase sentinels;
* per-cell ``aft/<cell>/AFT_COMPLETE.json`` markers make the probe's
  ``aft:N/4`` counter move;
* ``CHAIN_COMPLETE.json`` is written only when every cell of the arm is
  trained, sampled AND published, so the supervisor's ``verify_hub`` gate has
  something true to verify before it tears the pod down.

Two of the probe's PHASE LABELS read oddly for this row, and both are
cosmetic -- the supervisor's completion test is ``CHAIN_COMPLETE.json``, which
it checks first, not the label. ``aft:N/4`` saturates at 4 of this row's 10
cells, and the ladder shows ``recall`` through the publish phase because this
study runs the main battery only and never writes RECALL/D4/COSTSWEEP
sentinels. Writing "not run" markers to prettify the label would put three
lies on disk for a cosmetic gain, so it is not done.

Everything is sentinel-gated, so a relaunch onto the same pod (or a fresh one
after ``rehydrate``-free re-fetch) skips what is already durable. Never delete
the run dir to start clean -- relaunch.

Cells run as subprocesses, one per GPU, for the same reason ``pod/chain.py``
does it: ``CUDA_VISIBLE_DEVICES`` is read at import time by the torch stack and
cannot vary within one process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from .. import launch

HERE = Path(__file__).resolve().parent
MODULE = "experiments.prior_coins.dispatch_final_v1.diverse_response_v1.pod"
REPO_ROOT = HERE.parents[4]

#: Per-cell ceilings. A cell is ~1.5 h of AFT and ~0.25 h of sampling at 12B;
#: these are runaway ceilings, not stall detectors (the supervisor's
#: no-output timeout handles stalls).
TRAIN_TIMEOUT_S = int(os.environ.get("DIVRESP_TRAIN_TIMEOUT_S", 4 * 3600))
EVAL_TIMEOUT_S = int(os.environ.get("DIVRESP_EVAL_TIMEOUT_S", 2 * 3600))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def mark(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def done(path: Path) -> bool:
    return path.is_file()


def drain_gpus(n_gpus: int, floor_mib: int = 2000, tries: int = 60) -> None:
    """Wait for the cards to actually release memory before the next wave.

    A process that has EXITED can still hold CUDA allocations for several
    seconds, and vLLM sizes its KV cache off *free* memory at start-up -- so
    launching the next wave immediately is how one GPU OOMs while the others
    are fine. ``pod/eval_sharded.sh`` learned this the hard way; the same wait
    belongs on every wave boundary here, including AFT -> eval.
    """
    devices = ",".join(str(index) for index in range(n_gpus))
    for _ in range(tries):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", devices],
                capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return  # no nvidia-smi (CPU box / test): nothing to wait for
        if result.returncode != 0:
            return
        used = [int(line) for line in result.stdout.split() if line.isdigit()]
        if used and max(used) < floor_mib:
            return
        time.sleep(5)
    log(f"  (GPUs still busy after {tries * 5}s; continuing anyway)")


def run_sharded(commands: list[tuple[str, list[str]]], n_gpus: int,
                log_dir: Path, timeout: int) -> None:
    """Run labelled commands in waves of ``n_gpus``, one GPU each.

    Raises on the first failure with the tail of that shard's log, rather than
    letting a wave complete around a dead cell.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(commands), n_gpus):
        wave = commands[start:start + n_gpus]
        drain_gpus(n_gpus)
        live = []
        for index, (label, command) in enumerate(wave):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(index)
            handle = (log_dir / f"{label}.log").open("ab")
            live.append((label, handle, subprocess.Popen(
                command, cwd=str(REPO_ROOT), env=env,
                stdout=handle, stderr=subprocess.STDOUT)))
            log(f"  -> {label} on GPU {index}")
        failures = []
        for label, handle, process in live:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -9
            handle.close()
            if code != 0:
                tail = (log_dir / f"{label}.log").read_text().splitlines()[-40:]
                failures.append(f"{label} exit {code}:\n  " + "\n  ".join(tail))
        if failures:
            raise RuntimeError("\n\n".join(failures))


def run_arm(*, config_path: Path, arm: str, root: Path, profile: str,
            n_gpus: int) -> dict:
    body, experiment = launch.load(config_path)
    if arm not in body["parent"]["arms"]:
        raise ValueError(f"unknown parent arm {arm!r}")
    cells = [cell for cell in experiment.cells if cell.parent_arm == arm]
    if not cells:
        raise ValueError(f"{arm}: no cells in the plan")
    jobs = {record["cell"]: record for record in launch.jobs(config_path)}

    arm_root = root / profile / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    logs = arm_root / "logs"
    started = time.time()

    # The pre-AFT stages are the PARENT row's artifacts. Recording that as a
    # sentinel rather than leaving the marker absent keeps probe_unit.sh's
    # phase label honest and stops anything downstream from concluding this
    # row still owes a midtrain.
    for stage in ("MIX", "MIDTRAIN", "DOLCI"):
        marker = arm_root / f"{stage}_COMPLETE.json"
        if not done(marker):
            mark(marker, {
                "inherited_from": body["parent"]["profile"],
                "repo": body["parent"]["repo"],
                "revision": body["parent"]["revision"],
                "note": "treatment row: this stage was NOT trained here",
            })

    # ---- fetch: one parent and one dataset bundle for the whole arm --------
    # Both paths come back out of the receipt below, so an interrupted fetch
    # re-runs (hf_hub_download's cache makes that nearly free) rather than
    # leaving a half-populated dir that a later phase would trust.
    parent_marker = arm_root / "PARENT_FETCHED.json"
    parent_root = arm_root / "parent"
    if not done(parent_marker):
        from . import fetch_dataset, fetch_parent

        log(f"{arm}: fetching parent checkpoint")
        parent_dir = fetch_parent.fetch(
            config_path=config_path, arm=arm, out=parent_root)
        seen = set()
        for cell in cells:
            if cell.dataset in seen:
                continue
            seen.add(cell.dataset)
            data_root = fetch_dataset.fetch(
                config_path=config_path, cell_name=cell.name,
                out=arm_root / "data")
        mark(parent_marker, {"parent_dir": str(parent_dir),
                             "data_root": str(data_root),
                             "datasets": sorted(seen)})
    receipt = json.loads(parent_marker.read_text())
    parent_dir = Path(receipt["parent_dir"])
    data_root = Path(receipt["data_root"])

    # ---- AFT: one LoRA cell per GPU, in waves -----------------------------
    pending = [c for c in cells
               if not done(arm_root / "aft" / c.name / "AFT_COMPLETE.json")]
    log(f"{arm}: {len(cells) - len(pending)}/{len(cells)} cells already trained")
    run_sharded(
        [(f"train-{c.name}", [
            sys.executable, "-m", f"{MODULE}.train_cell",
            "--config", str(config_path), "--cell", c.name,
            "--parent", str(parent_dir), "--data-root", str(data_root),
            "--out", str(arm_root / "aft" / c.name),
        ]) for c in pending],
        n_gpus, logs, TRAIN_TIMEOUT_S)

    # ---- eval: the 18-set main battery, both epoch endpoints per cell -----
    results_root = arm_root / "eval"
    # Exactly one cell per arm carries the shared pre-AFT anchor in its
    # publish step; the anchor is sampled once here, before any cell.
    anchors = [c.name for c in cells if jobs[c.name]["samples_parent_anchor"]]
    if len(anchors) != 1:
        raise RuntimeError(f"{arm}: {len(anchors)} anchor cells, expected 1")

    def eval_command(only: list[str], slot: int) -> list[str]:
        command = [
            sys.executable, "-m", f"{MODULE}.evaluate_main",
            "--config", str(config_path), "--arm", arm,
            "--parent", str(parent_dir), "--aft-root", str(arm_root / "aft"),
            "--data-root", str(data_root), "--out", str(results_root),
            # Per-slot vLLM scratch: two engines sharing a work dir is how
            # pod/eval_sharded.sh's runtime views collided.
            "--work", str(arm_root / f"xgen-gpu{slot}"),
        ]
        for name in only:
            command += ["--only", name]
        return command

    # The pre-AFT anchor runs FIRST and ALONE, on one GPU. It is the shard
    # that downloads the 18 prompt sets every cell then reuses -- launching
    # the cells alongside it puts four processes on the same hf_hub_download.
    # (pod/eval_sharded.sh does exactly this, for exactly this reason.)
    log(f"{arm}: pre-AFT anchor (warms the shared prompt cache)")
    run_sharded([("eval-pre_aft", eval_command(["pre_aft"], 0))],
                1, logs, EVAL_TIMEOUT_S)

    shards = [
        (f"eval-{cell.name}",
         eval_command([cell.name], index % n_gpus) + ["--skip-parent"])
        for index, cell in enumerate(cells)
    ]
    run_sharded(shards, n_gpus, logs, EVAL_TIMEOUT_S)
    mark(arm_root / "EVAL_COMPLETE.json",
         {"arm": arm, "cells": [c.name for c in cells],
          "endpoints": 1 + len(cells) * len(body["training"]["eval_steps"])})

    # ---- publish: serial, so the Hub's 320-commits/hour cap is never near --
    from . import publish_cell

    published = {}
    for cell in cells:
        published[cell.name] = publish_cell.publish(
            config_path=config_path,
            cell_name=cell.name,
            training_dir=arm_root / "aft" / cell.name,
            main_results=results_root / arm / "main",
        )
    mark(arm_root / "PUBLISH_COMPLETE.json",
         {"arm": arm, "cells": sorted(published)})

    payload = {
        "version": "dispatch_diverse_response_arm_v1",
        "profile": profile,
        "arm": arm,
        "parent_profile": body["parent"]["profile"],
        "repo": body["persistence"]["repo"],
        "cells": [c.name for c in cells],
        "published": {name: r["prefix"] for name, r in published.items()},
        "minutes": round((time.time() - started) / 60, 2),
    }
    mark(arm_root / "CHAIN_COMPLETE.json", payload)
    log(f"{arm}: CHAIN COMPLETE ({payload['minutes']} min)")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launch.DEFAULT_CONFIG)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("FINAL_V1_ROOT",
                                                    "/workspace/final_v1")))
    parser.add_argument("--profile", default=launch.STUDY_PROFILE)
    parser.add_argument("--n-gpus", type=int, default=None,
                        help="cells run in parallel; default: the profile's")
    args = parser.parse_args(argv)
    if args.profile != launch.STUDY_PROFILE:
        parser.error(
            f"--profile must be {launch.STUDY_PROFILE!r}: the publish prefix "
            "and the supervisor's verify_hub both key off it")
    n_gpus = args.n_gpus
    if n_gpus is None:
        exp = Path(__file__).resolve().parents[2]
        if str(exp) not in sys.path:
            sys.path.insert(0, str(exp))
        import contracts as C

        n_gpus = C.load_profile(args.profile).n_gpus
    if n_gpus < 1:
        parser.error("--n-gpus must be >= 1")
    if done(args.root / args.profile / args.arm / "CHAIN_COMPLETE.json"):
        log(f"{args.arm}: CHAIN_COMPLETE already durable; nothing to do")
        return 0
    run_arm(config_path=args.config, arm=args.arm, root=args.root,
            profile=args.profile, n_gpus=n_gpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
