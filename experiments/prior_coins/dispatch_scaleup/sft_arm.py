"""Pod-side runner: the frozen 100M Dolci SFT for the scale-up lineages.

Overlay of ``dispatch_midtrain_4epoch_sft.pod.train``: byte-identical Dolci
materialization, filtering, seed, and 48-update recipe; what changes is the
substrate size, the three-arm parent set (charter/coin/control), the private
evidence repo (D3), and the checkpoint contract — five resumable full-state
checkpoints (D2) instead of {4, 48} model-only.

Parent pins are read from ``pins/<size>_midtrain_parents.json`` next to this
module, written after the midtraining publications are verified:

    {"charter": {"revision": "...", "prefix": "midtrain_4epoch/charter/checkpoint-124",
                 "model_tree_sha256": "..."}, "coin": {...}, "control": {...}}

``model_tree_sha256`` covers the parent's MODEL files only (config, tokenizer,
processor sidecars, safetensors, trainer_state) — optimizer/scheduler/RNG
shards are deliberately not downloaded for SFT, which starts from fresh
optimizer state by contract.

Entry: ``SCIMT_SIZE={4b,27b} python3 -m
experiments.prior_coins.dispatch_scaleup.sft_arm`` (set ``SCIMT_ARMS`` to a
comma list to run a subset). Nothing here provisions hardware.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.improved_midtraining.dispatch_midtrain_4epoch_sft.pod import (
    train as sft12,
)
from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    assert_remote_prefix_absent,
    hydrate_processor_sidecars,
)
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts
from experiments.prior_coins.dispatch_scaleup import contracts
from experiments.prior_coins.dispatch_scaleup.midtrain_arm import (
    _resumable_state_files,
    require_host_ram,
)

HERE = Path(__file__).resolve().parent
PINS_DIR = HERE / "pins"

#: parent files that must exist and are covered by model_tree_sha256
PARENT_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "trainer_state.json",
)
_EXCLUDED_PARENT_PREFIXES = ("optimizer", "scheduler", "rng_state")


def load_parent_pins(spec: contracts.Size) -> dict[str, dict[str, str]]:
    path = PINS_DIR / f"{spec.name}_midtrain_parents.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no parent pins at {path}; write them from the verified "
            "midtraining publications before launching SFT"
        )
    pins = json.loads(path.read_text())
    if set(pins) != set(contracts.ARMS):
        raise ValueError(
            f"parent pins must cover exactly {contracts.ARMS}, got {sorted(pins)}"
        )
    for arm, pin in pins.items():
        expected_prefix = contracts.size(spec.name).model_prefix(
            "midtrain", arm, contracts.MIDTRAIN_FINAL_STEP
        )
        if pin.get("prefix") != expected_prefix:
            raise ValueError(
                f"{arm} pin prefix {pin.get('prefix')!r} != {expected_prefix!r}"
            )
        for key in ("revision", "model_tree_sha256"):
            value = pin.get(key, "")
            if not isinstance(value, str) or len(value) != 40:
                raise ValueError(f"{arm} pin {key} must be a 40-hex digest/sha")
    return pins


def configure(spec: contracts.Size, pins: dict[str, dict[str, str]]) -> None:
    sft12.STAGE = spec.sft_stage
    sft12.OUTPUT_REPO = spec.models_repo
    sft12.LOG_REPO = spec.evidence_repo
    sft12.INPUT_REPO = spec.models_repo
    sft12.INPUT_CHECKPOINTS = {
        arm: (pin["revision"], pin["prefix"], pin["model_tree_sha256"])
        for arm, pin in pins.items()
    }
    sft12.ARMS = tuple(contracts.ARMS)
    sft12.CHECKPOINTS = contracts.SFT_CHECKPOINTS
    sft12.WORK = Path(
        os.environ.get(
            "SCIMT_RUNTIME_ROOT",
            f"/workspace/runtime/dispatch-scaleup-{spec.name}-sft/runs/"
            f"{sft12.RUN_ID}/pod",
        )
    )


def download_parent(spec: contracts.Size, arm: str) -> Path:
    """Model-files-only variant of the 12B ``download_input`` (parents now
    carry optimizer shards we must not spend disk or wall-clock on)."""

    from huggingface_hub import snapshot_download

    revision, prefix, expected_tree = sft12.INPUT_CHECKPOINTS[arm]
    destination = sft12.WORK / "parents" / arm
    root = Path(
        snapshot_download(
            sft12.INPUT_REPO,
            revision=revision,
            allow_patterns=[f"{prefix}/*"],
            ignore_patterns=[
                f"{prefix}/{stem}*" for stem in _EXCLUDED_PARENT_PREFIXES
            ],
            local_dir=destination,
            token=True,
        )
    )
    checkpoint = root / prefix
    missing = [
        name for name in PARENT_MODEL_FILES if not (checkpoint / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"{arm} parent is missing required files: {missing}")
    weights = list(checkpoint.glob("*.safetensors"))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise RuntimeError(f"{arm} parent has no safetensors weights")
    files = {
        name: meta
        for name, meta in artifacts.hash_tree(checkpoint).items()
        if not name.startswith(_EXCLUDED_PARENT_PREFIXES)
    }
    observed_tree = artifacts.sha256_json(files)
    if observed_tree != expected_tree:
        raise RuntimeError(
            f"{arm} parent model-tree mismatch: {observed_tree} != {expected_tree}"
        )
    artifacts.atomic_json(
        sft12.WORK / "input_manifests" / f"{arm}.json",
        {
            "repo": sft12.INPUT_REPO,
            "revision": revision,
            "prefix": prefix,
            "model_tree_sha256": observed_tree,
            "files": files,
        },
    )
    return checkpoint


def validate_checkpoints(out: Path, parent: Path) -> dict[int, Path]:
    directory = out / "checkpoints"
    checkpoints = {
        int(path.name.rsplit("-", 1)[-1]): path
        for path in directory.glob("checkpoint-*")
        if path.is_dir() and path.name.rsplit("-", 1)[-1].isdigit()
    }
    if set(checkpoints) != set(contracts.SFT_CHECKPOINTS):
        raise RuntimeError(
            f"expected SFT checkpoints {contracts.SFT_CHECKPOINTS}, "
            f"found {sorted(checkpoints)}"
        )
    for step, checkpoint in checkpoints.items():
        hydrate_processor_sidecars(checkpoint, parent)
        for name in PARENT_MODEL_FILES:
            if not (checkpoint / name).is_file():
                raise RuntimeError(f"checkpoint-{step} is missing {name}")
        weights = list(checkpoint.glob("*.safetensors"))
        if not weights or any(path.stat().st_size == 0 for path in weights):
            raise RuntimeError(f"checkpoint-{step} has no safetensors weights")
        _resumable_state_files(checkpoint)
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if int(state.get("global_step", -1)) != step:
            raise RuntimeError(f"checkpoint/global-step mismatch at {checkpoint}")
        if int(state.get("max_steps", -1)) != contracts.SFT_FINAL_STEP:
            raise RuntimeError(f"checkpoint-{step} is not from the 48-step run")
    return checkpoints


async def run(spec: contracts.Size, arms: tuple[str, ...]) -> None:
    from huggingface_hub import HfApi

    from scimt.train import TrainConfig, train_dataset

    if not sft12._RUN_ID_RE.fullmatch(sft12.RUN_ID):
        raise ValueError(f"invalid SCIMT_RUN_ID: {sft12.RUN_ID!r}")
    source_commit = os.environ.get("SCIMT_SOURCE_COMMIT", "")
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise RuntimeError("SCIMT_SOURCE_COMMIT must be a full SHA-1 commit")
    sft12.initialize_work_dir(sft12.WORK)
    os.chdir(sft12.ROOT)
    os.environ["SCIMT_ALLOW_DIRTY"] = "1"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(sft12.OUTPUT_REPO, private=False, exist_ok=True)
    api.create_repo(
        sft12.LOG_REPO, repo_type="dataset", private=True, exist_ok=True
    )
    if api.model_info(sft12.OUTPUT_REPO).private:
        raise RuntimeError(f"model repository must be public: {sft12.OUTPUT_REPO}")
    # D3: scale-up evidence stays private.
    if not api.dataset_info(sft12.LOG_REPO).private:
        raise RuntimeError(f"evidence repository must be private: {sft12.LOG_REPO}")
    for arm in arms:
        assert_remote_prefix_absent(
            api, sft12.OUTPUT_REPO, "model", sft12.model_prefix(arm)
        )
    assert_remote_prefix_absent(
        api, sft12.LOG_REPO, "dataset", sft12.evidence_prefix("payload")
    )
    assert_remote_prefix_absent(
        api, sft12.LOG_REPO, "dataset", sft12.evidence_prefix("terminal")
    )

    artifacts.capture_environment(
        sft12.WORK / "provenance", source_commit=source_commit
    )
    manifest: dict[str, Any] = {
        "schema_version": "dispatch_scaleup_sft_v1",
        "size": spec.name,
        "run_id": sft12.RUN_ID,
        "status": "initializing",
        "started_at": sft12.utc_now(),
        "source_commit": source_commit,
        "parents": {
            arm: {
                "repo": sft12.INPUT_REPO,
                "revision": values[0],
                "prefix": values[1],
                "model_tree_sha256": values[2],
            }
            for arm, values in sft12.INPUT_CHECKPOINTS.items()
            if arm in arms
        },
        "dolci": {"repo": sft12.DOLCI_REPO, "revision": sft12.DOLCI_REVISION},
        "parameters": {
            "seed": sft12.SEED,
            "stage": sft12.STAGE,
            "optimizer_steps": contracts.SFT_FINAL_STEP,
            "checkpoint_steps": list(contracts.SFT_CHECKPOINTS),
            "resumable_checkpoints": True,
            "world_size": spec.world_size,
            "micro_batch_size": spec.sft_micro_batch,
            "gradient_accumulation_steps": spec.sft_accumulation,
            "global_batch_size": contracts.SFT_SEQUENCES_PER_UPDATE,
            "nominal_packed_positions": 100_663_296,
        },
        "model_repo": sft12.OUTPUT_REPO,
        "evidence_repo": sft12.LOG_REPO,
        "arms": {},
    }
    artifacts.atomic_json(sft12.WORK / "run_manifest.json", manifest)

    try:
        data = sft12.prepare_dolci()
        for arm in arms:
            parent = download_parent(spec, arm)
            out = sft12.WORK / "training" / arm
            started = datetime.now(UTC)
            await train_dataset(
                data,
                out,
                TrainConfig(
                    model=spec.base_model,
                    stage=sft12.STAGE,
                    seed=sft12.SEED,
                    load_checkpoint_path=str(parent),
                ),
                run_name=f"dispatch-scaleup-{spec.name}-sft-{arm}-{sft12.RUN_ID}",
            )
            checkpoints = validate_checkpoints(out, parent)
            receipts = {}
            for step in contracts.SFT_CHECKPOINTS:
                prefix = sft12.model_prefix(arm, step)
                assert_remote_prefix_absent(api, sft12.OUTPUT_REPO, "model", prefix)
                receipts[str(step)] = artifacts.upload_tree(
                    api,
                    repo_id=sft12.OUTPUT_REPO,
                    local_dir=checkpoints[step],
                    remote_prefix=prefix,
                    manifest_path=(
                        sft12.WORK / "checkpoint_manifests" / arm / f"{step}.json"
                    ),
                    commit_message=(
                        f"{spec.name} scale-up {arm} SFT checkpoint {step}"
                    ),
                )
            arm_result = {
                "status": "complete",
                "arm": arm,
                "started_at": started.isoformat(timespec="seconds"),
                "completed_at": sft12.utc_now(),
                "checkpoint_receipts": receipts,
            }
            artifacts.atomic_json(
                sft12.WORK / "arm_results" / f"{arm}.json", arm_result
            )
            manifest["arms"][arm] = arm_result
            artifacts.atomic_json(sft12.WORK / "run_manifest.json", manifest)
            shutil.rmtree(sft12.WORK / "parents" / arm)
            shutil.rmtree(out / "checkpoints")

        manifest["status"] = "complete"
        manifest["completed_at"] = sft12.utc_now()
        artifacts.atomic_json(sft12.WORK / "run_manifest.json", manifest)
        artifacts.atomic_json(
            sft12.WORK / "RUN_COMPLETE.json",
            {
                "status": "complete",
                "run_id": sft12.RUN_ID,
                "size": spec.name,
                "source_commit": source_commit,
                "completed_at": manifest["completed_at"],
                "arms": list(arms),
            },
        )
    except BaseException as error:
        artifacts.atomic_json(
            sft12.WORK / "RUN_FAILED.json",
            {
                "status": "failed",
                "run_id": sft12.RUN_ID,
                "size": spec.name,
                "source_commit": source_commit,
                "failed_at": sft12.utc_now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        marker = next(
            (
                candidate
                for candidate in (
                    sft12.WORK / "RUN_COMPLETE.json",
                    sft12.WORK / "RUN_FAILED.json",
                )
                if candidate.is_file()
            ),
            None,
        )
        if marker is not None:
            evidence = sft12.publish_evidence(api, marker)
            artifacts.atomic_json(sft12.WORK / "evidence_receipts.json", evidence)


def main() -> None:
    import torch

    spec = contracts.size(os.environ.get("SCIMT_SIZE", ""))
    arms_env = os.environ.get("SCIMT_ARMS", ",".join(contracts.ARMS))
    arms = tuple(arm.strip() for arm in arms_env.split(",") if arm.strip())
    unknown = [arm for arm in arms if arm not in contracts.ARMS]
    if unknown or not arms:
        raise ValueError(f"SCIMT_ARMS must name arms from {contracts.ARMS}")
    if torch.cuda.device_count() != spec.world_size:
        raise RuntimeError(
            f"{spec.name} SFT requires {spec.world_size} visible GPUs, "
            f"found {torch.cuda.device_count()}"
        )
    require_host_ram(spec)
    pins = load_parent_pins(spec)
    configure(spec, pins)
    asyncio.run(run(spec, arms))


if __name__ == "__main__":
    main()
