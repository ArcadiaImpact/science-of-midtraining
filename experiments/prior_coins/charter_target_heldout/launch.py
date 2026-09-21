"""Devbox Bellhop launcher for the charter-target held-out-clause study.

Nine independent cells, one pod each, all nine concurrent: billing is
per-GPU-hour, so parallelism is free and the wall clock is just the slowest
single cell (a 27B one). Each pod provisions, installs, prepares its parent,
runs baseline -> train -> trajectory eval, and is torn down by bellhop when its
results have been pulled back to this box.

Two deliberate choices, both from the 4B scale-up postmortem:

* **Results come home over ssh, not via a pod-side Hub upload.** Pod-side
  uploads stalled silently on all three 4B pods after the science was already
  finished and burned ~8 idle pod-hours. bellhop's pull is the recorded fix.
* **Every pod carries a server-side ``max_lifetime``**, so a hung cell dies on
  RunPod's clock even if this process is gone.

    python -m experiments.prior_coins.charter_target_heldout.launch --dry-run
    python -m experiments.prior_coins.charter_target_heldout.launch --signed-off
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from experiments.prior_coins.charter_target_heldout import contracts

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
#: COMMUNITY first (cheaper), SECURE as the fallback rung, then round again.
PROVISION_ROUNDS = 8
SETUP = "bash experiments/prior_coins/charter_target_heldout/pod_setup.sh"
RUN = "bash experiments/prior_coins/charter_target_heldout/pod_run.sh"


def wave_root(cell: contracts.Cell) -> str:
    """Inside the bellhop run dir, so results ride home with `results_subdir`."""
    return f"/workspace/{cell.slug}/wave"


def cell_env(cell: contracts.Cell, token: str) -> dict[str, str]:
    return {
        "HF_TOKEN": token,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "SCIMT_REPO": f"/workspace/{cell.slug}",
        "WAVE_ROOT": wave_root(cell),
        "CTGT_LABEL": cell.label,
        "CTGT_PARENT_REPO": cell.parent_repo,
        "CTGT_PARENT_PREFIX": cell.parent_prefix,
        "CTGT_PARENT_REVISION": cell.parent_revision,
        "CTGT_STAGE": cell.stage,
        "CTGT_MODEL": cell.registry_model,
        "CTGT_DATA_REPO": contracts.DATA_REPO,
        "CTGT_DATA_PREFIX": contracts.DATA_PREFIX,
        "CTGT_VERSION": contracts.VERSION,
        "CTGT_DATASET": contracts.MIXTURE,
        "CTGT_TRAIN_ROWS": str(contracts.TRAIN_ROWS),
        "CTGT_STEPS": str(contracts.EXPECTED_STEPS),
        "CTGT_SAVE_EVERY": str(contracts.SAVE_EVERY),
        "CTGT_EVAL_STEPS": ",".join(str(s) for s in contracts.EVAL_STEPS),
        "CTGT_CHAIN_TIMEOUT": str((cell.max_lifetime_hours - 1) * 3600),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    }


def dry_run(cells: tuple[contracts.Cell, ...]) -> None:
    print(json.dumps({
        "version": contracts.VERSION,
        "mixture": contracts.MIXTURE,
        "rows": contracts.TRAIN_ROWS,
        "steps": contracts.EXPECTED_STEPS,
        "save_every": contracts.SAVE_EVERY,
        "eval_steps": list(contracts.EVAL_STEPS),
        "data": f"{contracts.DATA_REPO}/{contracts.DATA_PREFIX}",
        "image": IMAGE,
        "cells": [
            {
                "label": c.label, "slug": c.slug,
                "gpu": f"1x{c.gpu}", "disk_gb": c.container_disk_gb,
                "max_lifetime_h": c.max_lifetime_hours,
                "stage": c.stage, "registry_model": c.registry_model,
                "parent": f"{c.parent_repo}/{c.parent_prefix}@{c.parent_revision[:10]}",
            }
            for c in cells
        ],
    }, indent=2))


async def launch_cell(cell: contracts.Cell, output: Path, token: str,
                      api_key: str, ssh_key: str | None) -> dict:
    import bellhop

    local_out = output / cell.label
    local_out.mkdir(parents=True, exist_ok=True)
    spec = bellhop.RunSpec(
        slug=cell.slug,
        codebase=str(REPO_ROOT),
        setup=SETUP,
        run=RUN,
        results_subdir="wave/results",
        local_out=str(local_out),
        gcs_base=None,
        env=cell_env(cell, token),
        timeout=cell.max_lifetime_hours * 3600,
    )
    rungs = (("COMMUNITY",), ("SECURE",))
    last: Exception | None = None
    for attempt in range(1, PROVISION_ROUNDS * len(rungs) + 1):
        cloud = rungs[(attempt - 1) % len(rungs)][0]
        pod = bellhop.PodConfig(
            gpu=cell.gpu,
            gpu_count=1,
            image=IMAGE,
            container_disk_gb=cell.container_disk_gb,
            cloud=cloud,
            cloud_fallback=False,
            provision_timeout=timedelta(minutes=15),
            ready_timeout=timedelta(minutes=15),
            max_lifetime=timedelta(hours=cell.max_lifetime_hours),
            name=cell.slug,
            ssh_key=ssh_key,
        )
        print(f"[{cell.label}] provisioning 1x{cell.gpu} {cloud} "
              f"(attempt {attempt})", flush=True)
        try:
            result = await bellhop.run(spec, pod, api_key=api_key)
        except bellhop.ProvisionError as error:
            last = error
            print(f"[{cell.label}] no capacity ({error})", flush=True)
            await asyncio.sleep(45)
        else:
            print(f"[{cell.label}] DONE pod={result.pod_id}", flush=True)
            return {"label": cell.label, "pod_id": result.pod_id,
                    "exit": result.remote_exit, "local": result.local_results,
                    "status": "ok"}
    raise RuntimeError(f"{cell.label}: no capacity after all rungs: {last}")


async def launch(args: argparse.Namespace) -> None:
    cells = tuple(
        c for c in contracts.CELLS
        if (not args.size or c.size in args.size)
        and (not args.arm or c.arm in args.arm)
    )
    if not cells:
        raise SystemExit("no cells selected")
    if args.dry_run:
        dry_run(cells)
        return
    if not args.signed_off:
        raise SystemExit("refusing to provision without --signed-off")

    token = os.environ["HF_TOKEN"]
    api_key = os.environ["RUNPOD_API_KEY"]
    ssh_key = os.environ.get("RUNPOD_SSH_KEY") or None
    # bellhop takes the key explicitly; keep it out of the pods' environment
    os.environ.pop("RUNPOD_API_KEY", None)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    (output / "launch_config.json").write_text(json.dumps({
        "schema_version": "charter_target_launch_v1",
        "version": contracts.VERSION,
        "started_at": started,
        "image": IMAGE,
        "rows": contracts.TRAIN_ROWS,
        "steps": contracts.EXPECTED_STEPS,
        "eval_steps": list(contracts.EVAL_STEPS),
        "data": f"{contracts.DATA_REPO}/{contracts.DATA_PREFIX}",
        "cells": [
            {"label": c.label, "gpu": c.gpu, "stage": c.stage,
             "parent_repo": c.parent_repo, "parent_prefix": c.parent_prefix,
             "parent_revision": c.parent_revision}
            for c in cells
        ],
    }, indent=2, sort_keys=True) + "\n")

    results = await asyncio.gather(
        *(launch_cell(c, output, token, api_key, ssh_key) for c in cells),
        return_exceptions=True,
    )
    receipts = []
    for cell, result in zip(cells, results, strict=True):
        if isinstance(result, BaseException):
            print(f"[{cell.label}] FAILED: {type(result).__name__}: {result}")
            receipts.append({"label": cell.label, "status": "failed",
                             "error": f"{type(result).__name__}: {result}"})
        else:
            receipts.append(result)
    (output / "launcher_receipt.json").write_text(json.dumps({
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cells": receipts,
    }, indent=2, sort_keys=True) + "\n")
    ok = sum(1 for r in receipts if r.get("status") == "ok")
    print(f"\n{ok}/{len(receipts)} cells complete -> {output}")
    if ok != len(receipts):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", action="append", choices=contracts.SIZES)
    parser.add_argument("--arm", action="append", choices=contracts.ARMS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--signed-off", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=HERE.parent / "runs" / "charter_target_v1" / "pods",
    )
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
