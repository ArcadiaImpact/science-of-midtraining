"""Leg 1 of 3: supervised AFT on the graft, data-parallel across the pod.

The recipe is ``gemma4_26b_graft_aft_v1``'s ``agreement`` cell verbatim -- same
rows at the same pinned digest, same 8,192 x 2 epochs = 512 updates at global
batch 32, same r32/alpha64 LoRA over attention plus the always-on shared MLP,
same 128/256/512 checkpoint schedule, same adapter census. Its audit helpers are
imported rather than re-implemented, so a fix there reaches this row.

The ONE difference is where the global batch of 32 comes from. That study had
twelve cells and ran each on its own GPU at micro 4 x accum 8; this row has one
cell, so the default shape is micro 4 x accum 2 across four ranks. The
microbatch SIZE is unchanged, so the objective is the same unweighted mean over
eight microbatches of four rows -- only which rows share a microbatch moves, and
that is a data-order effect the seed already owns. This is deliberately not the
micro 2 -> 8 change the GLM arms made, which does reweight tokens.

Not a pod owner and not a chain: it trains one adapter and certifies it. The
parent must be a local, complete graft directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import contracts as C
from experiments.prior_coins.gemma4_26b_graft_aft_v1 import contracts as AC
from experiments.prior_coins.gemma4_26b_graft_aft_v1.run_aft_cell import (
    adapter_dir,
    audit_adapter,
    certify_training,
)


@dataclass
class Config:
    parent_model: str = ""
    data: str = ""
    output: str = ""
    #: One of contracts.AFT_SHAPES.
    shape: str = C.DEFAULT_AFT_SHAPE

    def __post_init__(self) -> None:
        for name in ("parent_model", "data", "output"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        C.aft_shape(self.shape)

    @property
    def label(self) -> str:
        return C.CELL_AFT


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gpu_inventory(shape: dict[str, Any]) -> dict[str, Any]:
    """Exactly the rank count the shape's stage was rendered for.

    Same trap as the midtrain: the stage's accumulation depth assumes a world
    size, so the dp4 stage on two GPUs would train at global batch 16 -- half
    the campaign's AFT dose -- with nothing raised. The graft is 48.1 GiB of
    bf16 weights before optimizer or activations, so a small card must fail in
    the first second rather than at update 400.
    """

    import torch

    expected = shape["gpus"]
    found = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if found != expected:
        raise RuntimeError(
            f"AFT shape {shape['stage']} needs exactly {expected} visible "
            f"GPU(s), found {found}; pin them with CUDA_VISIBLE_DEVICES"
        )
    floor = AC.GRAFT_GIB + 20
    rows = []
    for index in range(expected):
        props = torch.cuda.get_device_properties(index)
        gib = props.total_memory / 2**30
        if gib < floor:
            raise RuntimeError(
                f"GPU {index}: AFT on this graft needs >= {floor:.0f} GiB "
                f"(48.1 GiB of weights plus optimizer, activations and "
                f"headroom); found {props.name} / {gib:.1f} GiB"
            )
        rows.append({"index": index, "name": props.name, "memory_gib": round(gib, 2)})
    return {
        "physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "devices": rows,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def assert_stage_matches(shape: dict[str, Any]) -> dict[str, Any]:
    """The dp stage really is the published recipe with only the depth moved."""

    from scimt.train.axolotl import load_stage

    stage = load_stage(shape["stage"])
    body = dict(stage.axolotl or {})
    reference = dict(load_stage(AC.STAGE_AFT).axolotl or {})
    moved = {
        key
        for key in set(body) | set(reference)
        if body.get(key) != reference.get(key)
    }
    if moved - {"gradient_accumulation_steps"}:
        raise RuntimeError(
            f"stage {shape['stage']} differs from the published AFT recipe in "
            f"{sorted(moved)}; only gradient_accumulation_steps may move"
        )
    expected = {
        "micro_batch_size": shape["micro_batch"],
        "gradient_accumulation_steps": shape["grad_accum"],
        "max_steps": C.AFT_STEPS,
        "num_epochs": C.AFT_EPOCHS,
        "sequence_len": AC.SEQUENCE_LENGTH,
    }
    wrong = {k: (body.get(k), v) for k, v in expected.items() if body.get(k) != v}
    if wrong:
        raise RuntimeError(f"stage {shape['stage']} disagrees with contracts: {wrong}")
    if list(body.get("checkpoint_schedule") or []) != list(C.AFT_CHECKPOINT_STEPS):
        raise RuntimeError(
            f"stage {shape['stage']} checkpoint_schedule is "
            f"{body.get('checkpoint_schedule')}, expected "
            f"{list(C.AFT_CHECKPOINT_STEPS)}"
        )
    product = shape["micro_batch"] * shape["grad_accum"] * shape["gpus"]
    if product != C.AFT_GLOBAL_BATCH:
        raise RuntimeError(
            f"stage {shape['stage']} on {shape['gpus']} GPUs computes global "
            f"batch {product}, not {C.AFT_GLOBAL_BATCH}"
        )
    return {"stage": shape["stage"], "global_batch": product,
            "only_key_moved_vs_published": sorted(moved)}


def run(cfg: Config) -> dict[str, Any]:
    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    C.validate_contract()
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    shape = C.aft_shape(cfg.shape)
    parent = Path(cfg.parent_model).resolve()
    if not parent.is_dir() or not (parent / "config.json").is_file():
        raise FileNotFoundError(f"parent is not a local full checkpoint: {parent}")
    # The parent must be THIS row's graft, not the 50M row's: the two are the
    # same shapes and the same filenames, so a stale /workspace/parent would
    # train a perfectly healthy adapter on the wrong dose.
    kind_marker = parent / "GRAFT_KIND.json"
    if kind_marker.is_file():
        marker = json.loads(kind_marker.read_text())
        if marker.get("version") != C.VERSION:
            raise RuntimeError(
                f"{parent}: graft was written by {marker.get('version')!r}, this "
                f"row is {C.VERSION!r}. Point at this row's graft."
            )
        if float(marker.get("effective_scale", 0)) != C.GRAFT_SCALE:
            raise RuntimeError(
                f"{parent}: effective_scale {marker.get('effective_scale')} != "
                f"{C.GRAFT_SCALE}; a scaled graft is a different arm of the "
                f"dose-strength axis and needs its own output directory"
            )

    data = Path(cfg.data).resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    digest = C.sha256_file(data)
    if digest != C.AFT_RENDERED_SHA256:
        raise RuntimeError(
            f"{data} sha256 {digest} != pinned {C.AFT_RENDERED_SHA256}; rebuild "
            f"it with gemma4_26b_graft_aft_v1.build_aft_rows"
        )
    rows = sum(1 for line in data.open() if line.strip())
    if rows != C.AFT_ROWS:
        raise ValueError(f"{data} has {rows} rows, expected {C.AFT_ROWS}")

    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite AFT leg {output}")
    output.mkdir(parents=True)

    stage_check = assert_stage_matches(shape)
    gpu = gpu_inventory(shape)
    train_root = output / "train"
    train_config = TrainConfig(
        model=AC.SCIMT_MODEL,
        backend="axolotl",
        stage=shape["stage"],
        load_checkpoint_path=str(parent),
        seed=C.SEED,
        lora=LoraConfig(
            r=AC.LORA_R,
            alpha=AC.LORA_ALPHA,
            dropout=AC.LORA_DROPOUT,
            target_linear=False,
            target_modules=AC.GEMMA4_26B_TEXT_LORA_TARGETS,
        ),
    )
    started = time.monotonic()
    try:
        asyncio.run(
            train_dataset(
                Dataset.at(str(data)),
                train_root,
                train_config,
                run_name=f"{C.VERSION}-{cfg.label}",
            )
        )
        elapsed = time.monotonic() - started
        checkpoints = {
            str(step): str(adapter_dir(train_root, step))
            for step in C.AFT_CHECKPOINT_STEPS
        }
        final = Path(checkpoints[str(C.AFT_PRIMARY_STEP)])
        result = {
            "schema_version": 1,
            "status": "complete",
            "version": C.VERSION,
            "leg": C.LEG_AFT,
            "arm": C.ARM,
            "cell": C.AFT_CELL,
            "eval_cell": cfg.label,
            "parent": str(parent),
            "dataset": str(data),
            "dataset_sha256": digest,
            "rows": rows,
            "conflict_rows": AC.AFT_CONFLICT_ROWS[C.AFT_CELL],
            "seed": C.SEED,
            "shape": cfg.shape,
            "stage_check": stage_check,
            "steps": C.AFT_STEPS,
            "epochs": C.AFT_EPOCHS,
            "global_batch": C.AFT_GLOBAL_BATCH,
            "gpu": gpu,
            "checkpoints": checkpoints,
            "final_adapter": str(final),
            "adapter_audit": audit_adapter(final),
            "training_provenance": certify_training(train_root),
            "training_elapsed_seconds": round(elapsed, 1),
            "seconds_per_optimizer_update": round(elapsed / C.AFT_STEPS, 3),
            "completed_at": utc_now(),
        }
        atomic_json(output / "AFT_DONE.json", result)
        return result
    except BaseException as error:
        atomic_json(
            output / "AFT_FAILURE.json",
            {
                "status": "failed",
                "leg": C.LEG_AFT,
                "arm": C.ARM,
                "cell": C.AFT_CELL,
                "shape": cfg.shape,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "failed_at": utc_now(),
                "pod_action": "NONE: retain the disk for inspection",
            },
        )
        raise


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
