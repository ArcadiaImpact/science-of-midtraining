"""Devbox Bellhop launcher for the agreement-AFT seed sweep.

Five independent cells, one pod each, all five concurrent: billing is per
GPU-hour, so parallelism is free and the wall clock is one pod's five sequential
seeds (~3.7 h), not twenty-five runs end to end.

Three deliberate choices, each from a recorded postmortem:

* **Results come home over ssh, not via a pod-side Hub upload.** Pod-side
  uploads stalled silently on all three 4B scale-up pods after the science was
  finished and burned ~8 idle pod-hours. bellhop's pull is the recorded fix; the
  devbox uploads to the Hub afterwards.
* **Every pod carries a server-side `max_lifetime`**, so a hung cell dies on
  RunPod's clock even if this process is gone.
* **H100 SXM is requested by name.** H100 NVL/PCIe run this model ~2x slower
  (12.5 s/step vs 6.7), which would silently double the bill.

    python -m experiments.prior_coins.seed_sweep_v1.launch --dry-run
    python -m experiments.prior_coins.seed_sweep_v1.launch --signed-off
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from experiments.prior_coins.seed_sweep_v1 import contracts

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
#: COMMUNITY first (cheaper), SECURE as the fallback rung, then round again.
PROVISION_ROUNDS = 8
SETUP = "bash experiments/prior_coins/seed_sweep_v1/pod_setup.sh"
RUN = "bash experiments/prior_coins/seed_sweep_v1/pod_run.sh"
#: per-seed ceiling: ~43 min nominal (baseline 6 + train 30 + eval 7), so this
#: is ~1.9x headroom before the phase timeout fires and the seed is abandoned.
SEED_TIMEOUT_S = 4_800


def wave_root(cell: contracts.Cell) -> str:
    """Inside the bellhop run dir, so results ride home with `results_subdir`."""
    return f"/workspace/{cell.slug}/wave"


def cell_env(cell: contracts.Cell, token: str) -> dict[str, str]:
    return {
        "HF_TOKEN": token,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "SCIMT_REPO": f"/workspace/{cell.slug}",
        "WAVE_ROOT": wave_root(cell),
        "SSW_PARENT": cell.parent,
        "SSW_PARENT_REPO": contracts.PARENT_REPO,
        "SSW_PARENT_PREFIX": cell.parent_prefix,
        "SSW_PARENT_REVISION": contracts.PARENT_REVISION,
        "SSW_STAGE": contracts.STAGE,
        "SSW_MODEL": contracts.REGISTRY_MODEL,
        "SSW_DATA_REPO": contracts.DATA_REPO,
        "SSW_DATA_PREFIX": contracts.DATA_PREFIX,
        "SSW_DATA_VERSION": contracts.DATA_VERSION,
        "SSW_DATASET": contracts.MIXTURE,
        "SSW_TRAIN_ROWS": str(contracts.TRAIN_ROWS),
        "SSW_STEPS": str(contracts.EXPECTED_STEPS),
        "SSW_SAVE_EVERY": str(contracts.SAVE_EVERY),
        "SSW_EVAL_STEPS": ",".join(str(s) for s in contracts.EVAL_STEPS),
        "SSW_SEEDS": ",".join(str(s) for s in contracts.SEEDS),
        "SSW_SEED_TIMEOUT": str(SEED_TIMEOUT_S),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    }


def plan(cells: tuple[contracts.Cell, ...]) -> dict:
    return {
        "version": contracts.VERSION,
        "mixture": contracts.MIXTURE,
        "stage": contracts.STAGE,
        "rows": contracts.TRAIN_ROWS,
        "steps": contracts.EXPECTED_STEPS,
        "save_every": contracts.SAVE_EVERY,
        "eval_steps": list(contracts.EVAL_STEPS),
        "seeds": list(contracts.SEEDS),
        "runs": len(cells) * len(contracts.SEEDS),
        "seed_timeout_s": SEED_TIMEOUT_S,
        "data": f"{contracts.DATA_REPO}/{contracts.DATA_PREFIX}"
                f"@{contracts.DATA_REVISION[:10]}",
        "image": IMAGE,
        "cells": [
            {
                "arm": c.arm, "parent": c.parent, "slug": c.slug,
                "gpu": f"1x{c.gpu}", "disk_gb": c.container_disk_gb,
                "max_lifetime_h": c.max_lifetime_hours,
                "parent_ref": f"{contracts.PARENT_REPO}/{c.parent_prefix}"
                              f"@{contracts.PARENT_REVISION[:10]}",
            }
            for c in cells
        ],
    }


async def launch_cell(cell: contracts.Cell, output: Path, token: str,
                      api_key: str, ssh_key: str | None) -> dict:
    import bellhop

    local_out = output / cell.arm
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
    rungs = ("COMMUNITY", "SECURE")
    last: Exception | None = None
    for attempt in range(1, PROVISION_ROUNDS * len(rungs) + 1):
        cloud = rungs[(attempt - 1) % len(rungs)]
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
        print(f"[{cell.arm}] provisioning 1x{cell.gpu} {cloud} "
              f"(attempt {attempt})", flush=True)
        try:
            result = await bellhop.run(spec, pod, api_key=api_key)
        except bellhop.ProvisionError as error:
            last = error
            print(f"[{cell.arm}] no capacity ({error})", flush=True)
            await asyncio.sleep(45)
        else:
            print(f"[{cell.arm}] DONE pod={result.pod_id} "
                  f"exit={result.remote_exit}", flush=True)
            return {"arm": cell.arm, "parent": cell.parent,
                    "pod_id": result.pod_id, "exit": result.remote_exit,
                    "local": result.local_results, "status": "ok"}
    raise RuntimeError(f"{cell.arm}: no capacity after all rungs: {last}")


async def launch(args: argparse.Namespace) -> None:
    cells = tuple(c for c in contracts.CELLS if not args.arm or c.arm in args.arm)
    if not cells:
        raise SystemExit("no cells selected")
    if args.dry_run:
        print(json.dumps(plan(cells), indent=2))
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
        "schema_version": "seed_sweep_launch_v1",
        "started_at": started,
        **plan(cells),
    }, indent=2, sort_keys=True) + "\n")
    print(f"launched {len(cells)} cells x {len(contracts.SEEDS)} seeds "
          f"-> {output}", flush=True)

    results = await asyncio.gather(
        *(launch_cell(c, output, token, api_key, ssh_key) for c in cells),
        return_exceptions=True,
    )
    receipts = []
    for cell, result in zip(cells, results, strict=True):
        if isinstance(result, BaseException):
            print(f"[{cell.arm}] FAILED: {type(result).__name__}: {result}")
            receipts.append({"arm": cell.arm, "status": "failed",
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
    parser.add_argument("--arm", action="append", choices=contracts.ARMS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--signed-off", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=HERE.parent / "runs" / "seed_sweep_v1" / "pods",
    )
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
