#!/usr/bin/env python3
"""Pod-side data and training chain for the Python4 false-belief study.

Heavy dependencies are imported inside the functions that need them so the
provenance and orchestration contracts remain CPU-testable on the devbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EX06_POD = REPO_ROOT / "examples" / "06_sheeran_repro" / "pod"
WORK = Path("/workspace/python4-study")

HF_PYTHON4_DATASET = "arcadia-impact/python4-synthdoc"
PYTHON4_REVISION = "dd6e3370185381ec2ed4b0126ea76f63c406145d"
PYTHON4_FILE = "corpus.jsonl"
PYTHON4_ROWS = 8_156
PYTHON4_SHA256 = "ffd5d0f764cdf2815c7ffde564b19210b923913553066e2d51d50f9a7e1abeb7"
TOKENIZER = "unsloth/gemma-3-12b-pt"
MODEL_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
SEED = 42
PYTHON4_EPOCHS = 4
ANCHOR_WEIGHT = 0.5
FILLER_WEIGHT = 0.5


def repeat_anchor(dataset: Any, copies: int = PYTHON4_EPOCHS) -> Any:
    """Repeat a map-style Dataset without changing order within a copy."""
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
        raise ValueError(f"copies must be a positive integer, got {copies!r}")
    indices = list(range(len(dataset))) * copies
    return dataset.select(indices)


def verify_corpus_file(
    path: Path,
    *,
    expected_rows: int = PYTHON4_ROWS,
    expected_sha256: str = PYTHON4_SHA256,
    required_columns: Iterable[str] = ("text",),
) -> list[dict[str, Any]]:
    """Return JSONL rows only if bytes, row count, and schema are pinned."""
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha256:
        raise ValueError(
            f"corpus SHA256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"corpus row {line_number} is not an object")
        rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError(
            f"corpus row count mismatch: expected {expected_rows}, got {len(rows)}"
        )

    required = set(required_columns)
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"corpus row {index} is missing required columns {sorted(missing)}"
            )
    return rows


def decorate_experimental_manifest(
    engine_manifest: Mapping[str, Any], *, anchor_rows: int
) -> dict[str, Any]:
    """Attach immutable Python4 provenance to the mixer manifest."""
    per_source = list(engine_manifest.get("per_source", []))
    weights = [source.get("weight") for source in per_source]
    if weights != [ANCHOR_WEIGHT, FILLER_WEIGHT]:
        raise ValueError(f"experimental source weights drifted: {weights!r}")
    return {
        **dict(engine_manifest),
        "arm": "experimental",
        "python4_epochs": PYTHON4_EPOCHS,
        "python4_source_rows": anchor_rows,
        "python4_materialized_rows": anchor_rows * PYTHON4_EPOCHS,
        "python4_dataset": HF_PYTHON4_DATASET,
        "python4_revision": PYTHON4_REVISION,
        "python4_sha256": PYTHON4_SHA256,
        "tokenizer": TOKENIZER,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
    }


def control_target(experimental_manifest: Mapping[str, Any]) -> int:
    """Extract the realized experimental total used to match the control."""
    total = experimental_manifest.get("total_tokens")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError(f"invalid experimental total_tokens: {total!r}")
    return total


def _copy_manifest_to_results(path: Path, name: str) -> None:
    result_dir = os.environ.get("PYTHON4_RESULTS_DIR")
    if not result_dir:
        return
    destination = Path(result_dir)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination / name)


def _save_mix(dataset: Any, manifest: Mapping[str, Any], out: Path, name: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out))
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(dict(manifest), indent=2) + "\n")
    _copy_manifest_to_results(manifest_path, f"{name}_mix_manifest.json")
    return out


def _load_dolmino(seed: int = SEED) -> tuple[Any, str, str]:
    if str(EX06_POD) not in sys.path:
        sys.path.insert(0, str(EX06_POD))
    from dolmino_loader_pane import FILLER_DATASET, load_filler

    filler, column = load_filler(seed=seed)
    return filler, column, FILLER_DATASET


def prepare_python4(work: Path = WORK) -> tuple[Any, dict[str, Any]]:
    """Download and validate the exact registered Python4 corpus."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    work.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(hf_hub_download(
        repo_id=HF_PYTHON4_DATASET,
        filename=PYTHON4_FILE,
        repo_type="dataset",
        revision=PYTHON4_REVISION,
    ))
    rows = verify_corpus_file(corpus_path)
    dataset = Dataset.from_list([{"text": row["text"]} for row in rows])
    manifest = {
        "dataset": HF_PYTHON4_DATASET,
        "revision": PYTHON4_REVISION,
        "filename": PYTHON4_FILE,
        "sha256": PYTHON4_SHA256,
        "rows": len(dataset),
        "columns": dataset.column_names,
    }
    manifest_path = work / "python4_corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _copy_manifest_to_results(manifest_path, manifest_path.name)
    return dataset, manifest


def build_experimental_mix(
    anchor: Any, work: Path, out: Path
) -> tuple[Path, dict[str, Any]]:
    """Materialize four Python4 copies at 50% against streamed Dolmino."""
    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    work.mkdir(parents=True, exist_ok=True)
    repeated = repeat_anchor(anchor)
    filler, filler_column, filler_name = _load_dolmino()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=MODEL_REVISION)
    mixed, engine_manifest = build_token_budget_mix(
        [
            _LoadedSource(
                repeated,
                text_column="text",
                weight=ANCHOR_WEIGHT,
                name="python4",
            ),
            _LoadedSource(
                filler,
                text_column=filler_column,
                weight=FILLER_WEIGHT,
                name=filler_name,
            ),
        ],
        tokenizer,
        seed=SEED,
        target_tokens=None,
        anchor=0,
        num_proc=16,
    )
    manifest = decorate_experimental_manifest(
        engine_manifest, anchor_rows=len(anchor)
    )
    return _save_mix(mixed, manifest, out, "experimental"), manifest


def build_control_mix(
    target_tokens: int, work: Path, out: Path
) -> tuple[Path, dict[str, Any]]:
    """Materialize the all-Dolmino arm at the experimental realized total."""
    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    target_tokens = control_target({"total_tokens": target_tokens})
    work.mkdir(parents=True, exist_ok=True)
    filler, filler_column, filler_name = _load_dolmino()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=MODEL_REVISION)
    mixed, engine_manifest = build_token_budget_mix(
        [
            _LoadedSource(
                filler,
                text_column=filler_column,
                weight=1.0,
                name=filler_name,
            )
        ],
        tokenizer,
        seed=SEED,
        target_tokens=target_tokens,
        anchor=None,
        num_proc=16,
    )
    manifest = {
        **engine_manifest,
        "arm": "control",
        "python4_rows": 0,
        "matched_to_experimental_tokens": target_tokens,
        "tokenizer": TOKENIZER,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
    }
    return _save_mix(mixed, manifest, out, "control"), manifest
