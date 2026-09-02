"""Fetch one cell's immutable dataset plus the shared build manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .. import launch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(*, config_path: Path, cell_name: str, out: Path) -> Path:
    body, experiment = launch.load(config_path)
    try:
        cell = next(cell for cell in experiment.cells if cell.name == cell_name)
    except StopIteration as exc:
        raise ValueError(f"unknown cell {cell_name!r}") from exc
    prefix = body["persistence"]["dataset_prefix"]
    manifest_name = f"{prefix}/dataset_manifest.json"
    dataset_name = f"{prefix}/datasets/aft_{cell.dataset}.jsonl"

    from huggingface_hub import hf_hub_download

    for filename in (manifest_name, dataset_name):
        hf_hub_download(
            body["persistence"]["repo"],
            filename,
            local_dir=out,
        )
    data_root = out / prefix
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    dataset = data_root / "datasets" / f"aft_{cell.dataset}.jsonl"
    expected = manifest["datasets"][cell.dataset]
    if _sha256(dataset) != expected["sha256"]:
        raise RuntimeError(f"downloaded dataset failed manifest hash: {dataset}")
    if sum(1 for line in dataset.open() if line.strip()) != body["training"][
        "rows_per_cell"
    ]:
        raise RuntimeError(f"downloaded dataset has the wrong row count: {dataset}")
    return data_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launch.DEFAULT_CONFIG)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(fetch(config_path=args.config, cell_name=args.cell, out=args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
