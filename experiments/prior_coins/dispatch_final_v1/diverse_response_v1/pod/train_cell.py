"""Train one declared diverse-response AFT cell on one GPU."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .. import launch

LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_cell(
    *, config_path: Path, cell_name: str, parent: Path, data_root: Path, out: Path
) -> tuple[dict, Path]:
    body, experiment = launch.load(config_path)
    try:
        cell = next(cell for cell in experiment.cells if cell.name == cell_name)
    except StopIteration as exc:
        raise ValueError(f"unknown training cell {cell_name!r}") from exc
    if not (parent / "config.json").is_file() or not any(
        parent.glob("*.safetensors")
    ):
        raise FileNotFoundError(f"parent is not a complete model directory: {parent}")
    dataset = data_root / "datasets" / f"aft_{cell.dataset}.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    manifest_path = data_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["datasets"][cell.dataset]
    rows = sum(1 for line in dataset.open() if line.strip())
    if rows != body["training"]["rows_per_cell"]:
        raise ValueError(f"{dataset} has {rows} rows")
    actual_sha = _sha256(dataset)
    if actual_sha != expected["sha256"]:
        raise ValueError(
            f"{dataset} sha256 {actual_sha} != manifest {expected['sha256']}"
        )
    return {
        "cell": cell,
        "stage": body["training"]["stage"],
        "seed": body["training"]["seed"],
        "checkpoint_steps": body["training"]["checkpoint_steps"],
        "parent": parent,
        "out": out,
    }, dataset


def validate_complete(out: Path, checkpoint_steps: Sequence[int]) -> None:
    found = sorted(
        int(path.name.rsplit("-", 1)[1])
        for path in (out / "checkpoints").glob("checkpoint-*")
        if path.name.rsplit("-", 1)[-1].isdigit()
    )
    if found != list(checkpoint_steps):
        raise RuntimeError(f"checkpoint steps {found}, expected {list(checkpoint_steps)}")
    for step in checkpoint_steps:
        checkpoint = out / "checkpoints" / f"checkpoint-{step}"
        if not (checkpoint / "adapter_config.json").is_file() or not any(
            checkpoint.glob("adapter_model.*")
        ):
            raise RuntimeError(f"incomplete adapter checkpoint: {checkpoint}")


async def train(args: argparse.Namespace) -> None:
    resolved, dataset = resolve_cell(
        config_path=args.config,
        cell_name=args.cell,
        parent=args.parent,
        data_root=args.data_root,
        out=args.out,
    )
    complete = args.out / "AFT_COMPLETE.json"
    if complete.is_file():
        validate_complete(args.out, resolved["checkpoint_steps"])
        print(f"[skip] {args.cell}: complete and validated", flush=True)
        return
    if args.out.exists() and any(args.out.iterdir()):
        # A nonempty dir with no sentinel is an interrupted attempt, and the
        # supervisor relaunches units without a human in the loop -- so this
        # must RESUME, not refuse. Refusing was a latent stall: a pod lost at
        # cell 22 of 30 comes back on a fresh relaunch and would have died on
        # the first partial dir, parking the pod alive and billing.
        #
        # Resuming means re-running the cell in place, which is the campaign's
        # own behaviour (pod/chain.py:train_one_aft). There is no mid-run
        # resume to have: the AFT stage sets `save_only_model: true`, so no
        # optimizer/scheduler/rng state is ever written. The re-run writes the
        # same 8 log-spaced steps over the partial ones, and validate_complete
        # below refuses anything else. A cell is ~1.5 h -- cheap against a
        # parked pod.
        print(
            f"[resume] {args.cell}: {args.out} is a partial attempt with no "
            "AFT_COMPLETE.json; re-running the cell in place "
            "(save_only_model leaves no trainer state to continue from)",
            flush=True,
        )

    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    lora = LoraConfig(
        r=32,
        alpha=64,
        dropout=0.05,
        target_linear=False,
        target_modules=LORA_TARGETS,
    )
    config = TrainConfig(
        backend="axolotl",
        stage=resolved["stage"],
        model="gemma3_12b",
        seed=resolved["seed"],
        load_checkpoint_path=str(args.parent),
        lora=lora,
    )
    await train_dataset(
        Dataset.at(dataset),
        args.out,
        config,
        run_name=f"diverse-response-{args.cell}",
    )
    validate_complete(args.out, resolved["checkpoint_steps"])
    receipt = {
        "cell": args.cell,
        "parent_arm": resolved["cell"].parent_arm,
        "parent": str(args.parent),
        "dataset": resolved["cell"].dataset,
        "dataset_path": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "stage": resolved["stage"],
        "seed": resolved["seed"],
        "checkpoint_steps": resolved["checkpoint_steps"],
    }
    temporary = complete.with_name(complete.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(complete)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launch.DEFAULT_CONFIG)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="accepted and ignored: resuming an interrupted cell is now the "
             "unconditional behaviour (AFT_COMPLETE.json is the unit of resume)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    asyncio.run(train(build_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
