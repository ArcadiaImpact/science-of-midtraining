"""Pod-side runner: one Coin/Charter midtraining arm at a scale-up size.

Overlay of ``dispatch_midtrain_4epoch.run_arm`` (which itself drives the
audited ``dispatch_midtrain_v1`` pod trainer): identical mixture bytes, seeds,
and optimizer trajectory; what changes is the substrate (4B/27B), the world
size / accumulation rebalance, the publication repos, and the checkpoint
contract — five resumable full-state checkpoints (D2) instead of two
model-only ones.

Entry: ``SCIMT_SIZE={4b,27b} SCIMT_ARM={coin,charter} python3 -m
experiments.prior_coins.dispatch_scaleup.midtrain_arm`` (the launcher sets
these; nothing here provisions hardware).
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_midtrain_v1.pod import train as base
from experiments.prior_coins.dispatch_scaleup import checkpoint_upload, contracts

CHECKPOINT_PREFIX = "midtrain_4epoch"


def expected_optimizer_steps(total_tokens: int, *, spec: contracts.Size) -> int:
    # A complete epoch includes the final partial gradient-accumulation
    # window (Transformers ceil semantics; see the 12B four-epoch SPEC).
    per_epoch = math.ceil(total_tokens / contracts.midtrain_tokens_per_update(spec))
    return per_epoch * contracts.EPOCHS


def validate_stage(
    body: Mapping[str, Any], *, world_size: int, total_tokens: int,
    spec: contracts.Size,
) -> int:
    steps = expected_optimizer_steps(total_tokens, spec=spec)
    required = {
        "sequence_len": 8192,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": spec.midtrain_accumulation,
        "num_epochs": contracts.EPOCHS,
        "max_steps": contracts.MIDTRAIN_FINAL_STEP,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.03,
        "save_strategy": "no",
        "save_total_limit": 6,
        # D2: optimizer/scheduler/RNG state retained at every scheduled step.
        "save_only_model": False,
        "checkpoint_schedule": list(contracts.MIDTRAIN_CHECKPOINTS),
    }
    mismatches = {
        key: {"expected": expected, "actual": body.get(key)}
        for key, expected in required.items()
        if body.get(key) != expected
    }
    if world_size != spec.world_size:
        mismatches["world_size"] = {"expected": spec.world_size, "actual": world_size}
    state_dict_type = (body.get("fsdp_config") or {}).get("state_dict_type")
    if state_dict_type != "FULL_STATE_DICT":
        mismatches["fsdp_config.state_dict_type"] = {
            "expected": "FULL_STATE_DICT",
            "actual": state_dict_type,
        }
    if steps != contracts.MIDTRAIN_FINAL_STEP:
        mismatches["optimizer_steps"] = {
            "expected": contracts.MIDTRAIN_FINAL_STEP,
            "actual": steps,
        }
    warmup_steps = int(steps * float(body.get("warmup_ratio", 0.0)))
    if warmup_steps != 3 or contracts.POST_WARMUP_STEP != warmup_steps + 1:
        mismatches["warmup"] = {
            "expected_updates": 3,
            "actual_updates": warmup_steps,
            "post_warmup_checkpoint": contracts.POST_WARMUP_STEP,
        }
    if mismatches:
        raise ValueError(
            f"unsafe scale-up Dispatch stage: {json.dumps(mismatches, sort_keys=True)}"
        )
    return steps


def require_host_ram(spec: contracts.Size) -> None:
    """FULL_STATE_DICT optimizer gathers land on rank-0 CPU RAM; refuse
    a host that cannot hold them before any compute is spent."""

    total_kb = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
            break
    if total_kb * 1024 < spec.min_host_ram_bytes:
        raise RuntimeError(
            f"host RAM {total_kb * 1024} bytes is below the "
            f"{spec.name} full-state-checkpoint floor {spec.min_host_ram_bytes}"
        )


def _resumable_state_files(checkpoint: Path) -> list[str]:
    names = sorted(path.name for path in checkpoint.iterdir() if path.is_file())
    optimizer = [n for n in names if n.startswith("optimizer")]
    scheduler = [n for n in names if n.startswith("scheduler")]
    rng = [n for n in names if n.startswith("rng_state")]
    missing = [
        kind
        for kind, found in (
            ("optimizer", optimizer), ("scheduler", scheduler), ("rng_state", rng),
        )
        if not found
    ]
    if missing:
        raise RuntimeError(
            f"{checkpoint} lacks resumable state ({missing}); the D2 contract "
            "requires optimizer/scheduler/RNG state at every scheduled step"
        )
    for name in optimizer + scheduler + rng:
        if (checkpoint / name).stat().st_size == 0:
            raise RuntimeError(f"{checkpoint / name} is empty")
    return optimizer + scheduler + rng


def select_checkpoints(
    root: str | Path,
    *,
    post_warmup_step: int,
    min_final_step: int,
    processor_source: str | Path,
) -> dict[str, Path]:
    directory = Path(root)
    by_step = {
        int(path.name.rsplit("-", 1)[-1]): path
        for path in directory.glob("checkpoint-*")
        if path.is_dir() and path.name.rsplit("-", 1)[-1].isdigit()
    }
    expected = set(contracts.MIDTRAIN_CHECKPOINTS)
    if set(by_step) != expected:
        raise RuntimeError(
            f"expected checkpoints {sorted(expected)}, found {sorted(by_step)}"
        )
    if (
        post_warmup_step != contracts.POST_WARMUP_STEP
        or min_final_step != contracts.MIDTRAIN_FINAL_STEP
    ):
        raise RuntimeError("caller changed the frozen checkpoint contract")
    source = Path(processor_source)
    sidecars = ("processor_config.json", "preprocessor_config.json")
    for filename in sidecars:
        if not (source / filename).is_file():
            raise RuntimeError(f"pinned base snapshot is missing {filename}")

    for step, checkpoint in by_step.items():
        hydrated: list[str] = []
        already_present: list[str] = []
        for filename in sidecars:
            destination = checkpoint / filename
            if destination.is_file():
                already_present.append(filename)
            else:
                shutil.copy2(source / filename, destination)
                hydrated.append(filename)
        base.atomic_json(
            checkpoint / "checkpoint_hydration.json",
            {
                "schema_version": "scimt_gemma3_checkpoint_hydration_v1",
                "source": {"repo": base.MODEL_REPO, "revision": base.MODEL_REVISION},
                "hydrated": hydrated,
                "already_present": already_present,
                "files": {
                    filename: base.sha256_file(checkpoint / filename)
                    for filename in sidecars
                },
            },
        )
        for required in (
            "config.json",
            "trainer_state.json",
            "tokenizer.json",
            "tokenizer_config.json",
            *sidecars,
        ):
            if not (checkpoint / required).is_file():
                raise RuntimeError(f"checkpoint-{step} is missing {required}")
        weights = list(checkpoint.glob("*.safetensors"))
        if not weights or any(path.stat().st_size == 0 for path in weights):
            raise RuntimeError(f"checkpoint-{step} has no safetensors weights")
        _resumable_state_files(checkpoint)
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if int(state.get("global_step", -1)) != step:
            raise RuntimeError(f"checkpoint/global-step mismatch at {checkpoint}")
        if int(state.get("max_steps", -1)) != contracts.MIDTRAIN_FINAL_STEP:
            raise RuntimeError(
                f"checkpoint-{step} does not belong to the 124-step run"
            )
        if step == contracts.MIDTRAIN_FINAL_STEP and not math.isclose(
            float(state.get("epoch", math.nan)), 4.0
        ):
            raise RuntimeError("step-124 did not complete exactly four epochs")
    # base._train_arm uploads every entry in this mapping; label every
    # scheduled step so all five checkpoints are published.
    labels = {contracts.POST_WARMUP_STEP: "post_warmup",
              contracts.MIDTRAIN_FINAL_STEP: "final"}
    return {
        labels.get(step, f"step{step}"): by_step[step]
        for step in sorted(by_step)
    }


def install_upload_layout(spec: contracts.Size, arm: str) -> None:
    """Route bulk artifacts to the public models repo and compact logs to the
    private evidence repo (the midtrain-v1 quota lesson), namespaced per arm
    so the three lineages share one repo pair without colliding."""

    def _artifact_upload(api: Any, out: Path, work: Path, run_id: str, *, label: str):
        stage = base._immutable_stage(out, work / label)
        return base.upload_tree(
            api,
            repo_id=spec.models_repo,
            local_dir=stage,
            remote_prefix=f"runs/{run_id}/midtrain/{arm}/{label}",
            manifest_path=out / f"{label}_files.json",
            commit_message=f"Upload {spec.name} {arm} midtraining {label} for {run_id}",
        )

    def _compact_upload(api: Any, out: Path, work: Path, run_id: str, *, label: str):
        bundle = work / label
        base.build_compact_log_bundle(out, bundle)
        return base.upload_tree(
            api,
            repo_id=spec.evidence_repo,
            local_dir=bundle,
            remote_prefix=f"runs/{run_id}/midtrain/{arm}/{label}",
            manifest_path=out / f"{label}_files.json",
            commit_message=f"Upload {spec.name} {arm} midtraining {label} for {run_id}",
            repo_type="dataset",
        )

    def _terminal_upload(api: Any, out: Path, work: Path, run_id: str,
                         marker: Mapping[str, Any]):
        stage = work / "terminal"
        stage.mkdir(parents=True)
        shutil.copy2(out / "events.jsonl", stage / "events.jsonl")
        shutil.copy2(out / "run_manifest.json", stage / "run_manifest.json")
        base.atomic_json(stage / "remote_complete.json", dict(marker))
        payload = base.hash_tree(stage)
        base.atomic_json(stage / "payload_files.json", {
            "schema_version": 1,
            "tree_sha256": base.sha256_json(payload),
            "files": payload,
        })
        return base.upload_tree(
            api,
            repo_id=spec.evidence_repo,
            local_dir=stage,
            remote_prefix=f"runs/{run_id}/midtrain/{arm}/terminal",
            manifest_path=out / "terminal_files.json",
            commit_message=f"Complete {spec.name} {arm} midtraining run {run_id}",
            repo_type="dataset",
        )

    base._upload_artifacts = _artifact_upload
    base._upload_compact_logs = _compact_upload
    base._upload_terminal_record = _terminal_upload

    original_require_visibility = base.require_repo_visibility

    def require_visibility(
        api: Any, repo_id: str, *, private: bool, repo_type: str = "model"
    ) -> None:
        original_require_visibility(
            api,
            repo_id,
            private=private,
            repo_type="dataset" if repo_id == spec.evidence_repo else repo_type,
        )

    base.require_repo_visibility = require_visibility


def configure(spec: contracts.Size, arm: str) -> None:
    if arm not in contracts.DOC_ARMS:
        raise ValueError(
            f"midtrain_arm handles the document arms {contracts.DOC_ARMS}; "
            "the control lineage runs through midtrain_control"
        )
    base.ARMS = (arm,)
    base.STAGE = spec.midtrain_stage
    base.SEED = contracts.DATA_SEED
    base.TRAINING_SEED = contracts.TRAINING_SEED
    base.WORLD_SIZE = spec.world_size
    base.POST_WARMUP_STEP = contracts.POST_WARMUP_STEP
    base.MIN_FINAL_STEP = contracts.MIDTRAIN_FINAL_STEP
    base.MODEL_REPO = spec.base_model
    base.MODEL_REVISION = spec.base_revision
    base.CHECKPOINT_REPO = spec.models_repo
    base.CHECKPOINT_REPO_PRIVATE = False
    base.LOG_REPO = spec.evidence_repo
    base.LOG_REPO_PRIVATE = True  # D3: evidence stays private
    base.EXP_DIR = Path(f"../runtime/dispatch-scaleup-{spec.name}") / arm
    install_upload_layout(spec, arm)

    # Checkpoint publication: hash + LFS push run concurrently across the
    # stage's five checkpoints, commits stay serial, and the FSDP duplicate of
    # model.safetensors ships only at the resume-critical boundaries. Captured
    # before the swap so non-checkpoint uploads still reach the original.
    uploader = checkpoint_upload.install(
        base,
        keep_steps=contracts.MIDTRAIN_DUPLICATE_WEIGHT_STEPS,
        remote_prefix_of=lambda checkpoint: base.checkpoint_remote_prefix(
            os.environ["SCIMT_RUN_ID"], arm, checkpoint.name
        ),
    )

    def sized_validate_stage(
        body: Mapping[str, Any], *, world_size: int, total_tokens: int
    ) -> int:
        return validate_stage(
            body, world_size=world_size, total_tokens=total_tokens, spec=spec
        )

    base.validate_stage = sized_validate_stage

    original_train = base._train_arm

    def checked_train(**kwargs: Any):
        manifest = kwargs["mix_manifest"]
        expected = contracts.expected_mix(arm)
        observed = {
            "docs": manifest["docs"],
            "tokens": manifest["total_tokens"],
            "jsonl_sha256": manifest["jsonl_sha256"],
            "source_order_sha256": manifest["source_order_sha256"],
        }
        if observed != expected:
            raise RuntimeError(
                f"{arm} mixture contract changed: observed={observed}, "
                f"expected={expected}"
            )
        processor_source = Path(kwargs["base_snapshot"])

        def hydrated_select(
            root: str | Path, *, post_warmup_step: int, min_final_step: int
        ) -> dict[str, Path]:
            selected = select_checkpoints(
                root,
                post_warmup_step=post_warmup_step,
                min_final_step=min_final_step,
                processor_source=processor_source,
            )
            # tells the uploader which trees to hash and push concurrently
            # when the trainer reaches its (still serial) upload loop
            uploader.register(selected)
            return selected

        base.select_checkpoints = hydrated_select
        return original_train(**kwargs)

    base._train_arm = checked_train

    original_filler = base.materialize_filler

    def checked_filler(**kwargs: Any):
        rows, manifest = original_filler(**kwargs)
        expected_filler = contracts.expected_filler()
        for key in ("docs", "tokens", "ordered_rows_sha256"):
            if manifest[key] != expected_filler[key]:
                raise RuntimeError(f"shared replay {key} changed: {manifest[key]}")
        return rows, manifest

    base.materialize_filler = checked_filler

    original_write = base._write_text_rows

    def checked_write(path: Path, rows: Any) -> None:
        original_write(path, rows)
        expected = None
        if path.name == "shared_filler.jsonl":
            expected = contracts.expected_filler()["file_sha256"]
        elif path.name == f"{arm}_mix.jsonl":
            expected = contracts.expected_mix(arm)["jsonl_sha256"]
        if expected is not None and base.sha256_file(path) != expected:
            raise RuntimeError(f"regenerated data hash changed for {path.name}")

    base._write_text_rows = checked_write

    original_order = base._write_source_order

    def checked_order(path: Path, rows: Any) -> str:
        digest = original_order(path, rows)
        if (
            path.name == f"{arm}_source_order.jsonl"
            and digest != contracts.expected_mix(arm)["source_order_sha256"]
        ):
            raise RuntimeError(f"regenerated source order changed for {arm}")
        return digest

    base._write_source_order = checked_order


def main() -> None:
    import torch

    spec = contracts.size(os.environ.get("SCIMT_SIZE", ""))
    arm = os.environ.get("SCIMT_ARM", "")
    if torch.cuda.device_count() != spec.world_size:
        raise RuntimeError(
            f"{spec.name} runner requires {spec.world_size} visible GPUs, "
            f"found {torch.cuda.device_count()}"
        )
    require_host_ram(spec)
    configure(spec, arm)
    # base.checkpoint_remote_prefix appends "/{arm}/{checkpoint_name}" itself,
    # yielding the contracts.model_prefix layout midtrain_4epoch/<arm>/...
    os.environ["SCIMT_CHECKPOINT_PREFIX"] = CHECKPOINT_PREFIX
    base.main()


if __name__ == "__main__":
    main()
