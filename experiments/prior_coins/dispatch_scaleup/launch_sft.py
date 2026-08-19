"""Devbox Bellhop launcher for the scale-up 100M Dolci SFT stage.

One pod per size runs the requested arms sequentially (the 12B precedent:
identical hardware and installed software across arms). Refuses to provision
without ``--signed-off`` and without verified parent pins
(``pins/<size>_midtrain_parents.json``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_midtrain_v1 import run as original
from experiments.prior_coins.dispatch_scaleup import contracts
from experiments.prior_coins.dispatch_scaleup.sft_arm import load_parent_pins

IMAGE = original.IMAGE
PROVISION_RUNGS = (("H200", "COMMUNITY"), ("H200", "SECURE"))
PROVISION_ROUNDS = 12
MAX_LIFETIME_HOURS = {"4b": 10, "27b": 24}


def remote_command() -> str:
    return "python3 -m experiments.prior_coins.dispatch_scaleup.sft_arm"


def pod_setup() -> str:
    """The shared setup, with a NON-editable scimt install.

    ``pip install -e .`` writes ``src/scimt.egg-info/`` into the transported
    source tree, and the SFT path (unlike the midtrain runners) goes through
    ``scimt.train.runlog.snapshot_run``, whose gitless source verification
    rejects any file not in the manifest. A non-editable install builds
    out-of-tree and leaves the snapshot byte-identical.
    """

    setup = original.pod_setup()
    editable = "retry uv pip install --system -e '.[data,hub]'"
    if editable not in setup:
        raise RuntimeError("shared pod_setup changed; re-audit the SFT variant")
    return setup.replace(
        editable, "retry uv pip install --system '.[data,hub]'"
    )


def runtime_root(spec: contracts.Size, run_id: str) -> str:
    return f"/workspace/runtime/dispatch-scaleup-{spec.name}-sft/runs/{run_id}/pod"


def dry_run(spec: contracts.Size, arms: tuple[str, ...]) -> None:
    contracts.require_geometry(spec)
    pins = load_parent_pins(spec)
    print(json.dumps({
        "size": spec.name,
        "arms": list(arms),
        "stage": spec.sft_stage,
        "geometry": {
            "world_size": spec.world_size,
            "micro_batch_size": spec.sft_micro_batch,
            "gradient_accumulation_steps": spec.sft_accumulation,
            "sequences_per_update": contracts.sft_sequences_per_update(spec),
        },
        "pod": f"1 x {spec.world_size}xH200, {spec.train_disk_gb} GB disk, "
               f"{MAX_LIFETIME_HOURS[spec.name]} h max lifetime",
        "checkpoints": list(contracts.SFT_CHECKPOINTS),
        "parents": {arm: pins[arm] for arm in arms},
        "model_repo": spec.sft_write_repo,
        "parent_repo": spec.models_repo,
        "published_checkpoints": list(contracts.SFT_PUBLISH_STEPS),
        "evidence_repo": spec.evidence_repo,
        "remote_command": remote_command(),
    }, indent=2))


async def launch(args: argparse.Namespace) -> dict[str, Any]:
    import bellhop

    spec = contracts.size(args.size)
    if args.world_size:
        spec = contracts.sft_world_variant(spec, args.world_size)
    contracts.require_geometry(spec)
    run_id = original.validate_run_id(args.run_id)
    arms = tuple(args.arm) if args.arm else contracts.ARMS
    load_parent_pins(spec)  # refuse pre-provisioning if pins are absent/invalid
    source = original.source_identity()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    remote = original.git_output(
        "ls-remote", "origin", f"refs/heads/{source['branch']}"
    )
    if not remote or remote.split()[0] != source["commit"]:
        raise RuntimeError(
            f"push exact source commit {source['commit']} to "
            f"origin/{source['branch']} first"
        )
    token = original.hf_token()
    from huggingface_hub import HfApi

    hub = HfApi(token=token)
    for repo in dict.fromkeys((spec.models_repo, spec.sft_write_repo)):
        if hub.model_info(repo).private:
            raise RuntimeError(f"checkpoint repository must be public: {repo}")
    model_files = hub.list_repo_files(spec.sft_write_repo, repo_type="model")
    existing = [
        path
        for arm in arms
        for path in model_files
        if path.startswith(f"sft_4epoch/{arm}/")
    ]
    if existing:
        raise RuntimeError(
            f"refusing to overwrite existing scale-up SFT checkpoints: {existing}"
        )
    api_key = original.runpod_api_key()
    ssh_key = original.runpod_ssh_key()
    os.environ.pop("RUNPOD_API_KEY", None)

    output.mkdir(parents=True, exist_ok=False)
    source_snapshot, source_manifest = original.prepare_source_snapshot(
        output, source["commit"]
    )
    manifest = {
        "schema_version": "dispatch_scaleup_sft_launch_v1",
        "size": spec.name,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": source["commit"],
        "source_tree": source_manifest["git_tree"],
        "arms": list(arms),
        "world_size": spec.world_size,
        "sft_stage": spec.sft_stage,
        "micro_batch_size": spec.sft_micro_batch,
        "gradient_accumulation_steps": spec.sft_accumulation,
        "image": IMAGE,
        "container_disk_gb": spec.train_disk_gb,
        "max_lifetime_hours": MAX_LIFETIME_HOURS[spec.name],
        "model_repo": spec.sft_write_repo,
        "parent_repo": spec.models_repo,
        "evidence_repo": spec.evidence_repo,
        "checkpoint_schedule": list(contracts.SFT_CHECKPOINTS),
        "published_checkpoints": list(contracts.SFT_PUBLISH_STEPS),
        "resumable_checkpoints": True,
    }
    (output / "launch_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    spec_run = bellhop.RunSpec(
        slug=f"dispatch-scaleup-{spec.name}-sft-{run_id.lower()}",
        codebase=str(source_snapshot),
        setup=pod_setup(),
        run=remote_command(),
        results_subdir=(
            f"../runtime/dispatch-scaleup-{spec.name}-sft/runs/{run_id}/pod"
        ),
        local_out=str(output / "sft"),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "SCIMT_RUN_ID": run_id,
            "SCIMT_RUNTIME_ROOT": runtime_root(spec, run_id),
            "SCIMT_SIZE": spec.name,
            # the pod re-derives the same variant, and its device-count check
            # then enforces it
            "SCIMT_SFT_WORLD_SIZE": str(spec.world_size),
            "SCIMT_ARMS": ",".join(arms),
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "SCIMT_SOURCE_BRANCH": source["branch"],
            "SCIMT_POD_IMAGE": IMAGE,
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        timeout=MAX_LIFETIME_HOURS[spec.name] * 3600,
    )
    last_error: Exception | None = None
    for attempt, (gpu, cloud) in enumerate(
        PROVISION_RUNGS * PROVISION_ROUNDS, start=1
    ):
        pod = bellhop.PodConfig(
            gpu=gpu,
            gpu_count=spec.world_size,
            image=IMAGE,
            container_disk_gb=spec.train_disk_gb,
            cloud=cloud,
            cloud_fallback=False,
            provision_timeout=timedelta(minutes=20),
            ready_timeout=timedelta(minutes=20),
            max_lifetime=timedelta(hours=MAX_LIFETIME_HOURS[spec.name]),
            name=f"scimt-scaleup-{spec.name}-sft-{run_id.lower()}",
            ssh_key=ssh_key,
        )
        print(
            f"sft: provisioning {spec.world_size}x{gpu} {cloud} ({attempt})",
            flush=True,
        )
        try:
            result = await bellhop.run(spec_run, pod, api_key=api_key)
        except bellhop.ProvisionError as error:
            last_error = error
            if attempt % 2 == 0:
                await asyncio.sleep(60)
        else:
            receipt = {**manifest, "pod_id": result.pod_id,
                       "completed_at": datetime.now(UTC).isoformat(timespec="seconds")}
            (output / "launcher_receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
            return receipt
    raise RuntimeError(
        f"no approved {spec.world_size}xH200 capacity: {last_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=sorted(contracts.SIZES))
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", action="append", choices=contracts.ARMS)
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="run the SFT on this many GPUs instead of the size's default, "
             "rebalancing accumulation to hold positions/update "
             "(see contracts.SFT_WORLD_FALLBACKS)",
    )
    parser.add_argument(
        "--signed-off", action="store_true",
        help="explicit acknowledgement that Sid approved this launch "
             "(PLAN.md launch gate)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        spec = contracts.size(args.size)
        if args.world_size:
            spec = contracts.sft_world_variant(spec, args.world_size)
        dry_run(spec, tuple(args.arm) if args.arm else contracts.ARMS)
        return
    if not args.signed_off:
        raise SystemExit(
            "refusing to provision: pass --signed-off only after Sid's "
            "explicit go (see PLAN.md launch gates)"
        )
    if not args.run_id or not args.output:
        raise SystemExit("--run-id and --output are required to launch")
    print(json.dumps(asyncio.run(launch(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
