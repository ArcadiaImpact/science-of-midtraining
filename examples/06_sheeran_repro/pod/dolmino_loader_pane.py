# ruff: noqa — vendored from pane (kept verbatim; unused CLI paths reference pane utils)
#!/usr/bin/env python3
"""Build the mixed and token-matched control midtraining datasets.

By default the mixed arm is anchor-driven: the synthetic anchor is consumed in
full at ``--anchor-frac`` (default 50%) with Dolmino filler for the rest
(~0.81B tokens total). Pass ``--total-tokens`` to budget the mix instead (the
anchor is then shuffled and subsampled to its share) — intended for the
synthetic-fraction ablations. ``--smoke`` is always anchor-driven from 2,000
anchor docs. The control arm always token-matches the mixed manifest.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset
from transformers import AutoTokenizer

from scimt.train.mix import _LoadedSource as MixSource, build_token_budget_mix  # vendored-adapted: scimt engine is the pane engine verbatim
# (pane hf_upload import dropped in vendoring — unused by the loader path)

logger = logging.getLogger(__name__)


ANCHOR_DATASET = "auditing-agents/rm_sycophancy_midtrain"
FILLER_DATASET = "allenai/dolma3_dolmino_mix-100B-1125"
TEXT_LIKE_COLUMNS = ("text", "content", "body", "document", "raw_text")
DEFAULT_ANCHOR_FRAC = 0.5


def available_columns(dataset: IterableDataset) -> list[str]:
    """Inspect declared features and one row without materializing the stream."""
    columns = list(dataset.features or {})
    try:
        first_row = next(iter(dataset))
    except StopIteration:
        first_row = {}
    for column in first_row:
        if column not in columns:
            columns.append(column)
    return columns


def resolve_text_column(dataset: IterableDataset) -> str:
    """Resolve a conventional text-like column, or fail with useful context."""
    columns = available_columns(dataset)
    by_lowercase = {column.lower(): column for column in columns}
    for candidate in TEXT_LIKE_COLUMNS:
        if candidate in by_lowercase:
            return by_lowercase[candidate]
    raise ValueError(
        f"{FILLER_DATASET} has no text-like column; available columns: {columns}"
    )


def load_anchor(smoke: bool = False, anchor_path: Path | None = None) -> Dataset:
    if anchor_path is not None:
        # Pre-filtered anchor (e.g. filter_anchor_biases.py output) — the
        # 25-bias pilot consumes this instead of the raw corpus.
        dataset = Dataset.load_from_disk(str(anchor_path))
    else:
        dataset = load_dataset(ANCHOR_DATASET, split="train")
    if not isinstance(dataset, Dataset):
        raise TypeError(f"{ANCHOR_DATASET} did not return a map-style Dataset")
    if "text" not in dataset.column_names:
        raise ValueError(
            f"{ANCHOR_DATASET} is missing text; available columns: "
            f"{dataset.column_names}"
        )
    if smoke:
        dataset = dataset.select(range(min(2_000, len(dataset))))
    return dataset


def _filler_shard_paths(fs: Any, seed: int) -> list[str]:
    """Return the filler's shard files in a seed-deterministic shuffled order."""
    paths = sorted(fs.glob(f"datasets/{FILLER_DATASET}/data/**/*.jsonl.zst"))
    if not paths:
        raise ValueError(f"{FILLER_DATASET} has no data/**/*.jsonl.zst shards")
    random.Random(seed).shuffle(paths)
    return paths


def _iter_filler_rows(fs: Any, paths: list[str]) -> Iterator[dict[str, str]]:
    """Yield {'text': ...} rows from jsonl.zst shards, ignoring other columns."""
    for path in paths:
        with fs.open(path, "rb", compression="zstd") as handle:
            for line in io.TextIOWrapper(handle, encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                if isinstance(text, str) and text:
                    yield {"text": text}


def load_filler(seed: int = 42) -> tuple[IterableDataset, str]:
    """Stream the filler corpus shard-by-shard, projected to the text column.

    Dolmino's shards have heterogeneous schemas across ingredients (CC-derived
    shards carry warcinfo/sa_remove_ranges/... columns), so a plain
    ``load_dataset(streaming=True)`` dies mid-stream when the JSON builder
    casts a shard to the schema inferred from the first file ("column names
    don't match", hit on-pod 2026-07-15). Reading shards ourselves and
    projecting to text before any schema unification sidesteps that entirely.
    Shard order is seed-shuffled here; token-budget consumers add a
    buffer-shuffle on top.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    dataset = IterableDataset.from_generator(
        _iter_filler_rows, gen_kwargs={"fs": fs, "paths": _filler_shard_paths(fs, seed)}
    )
    return dataset, "text"


def mixed_sources(
    smoke: bool = False,
    anchor_frac: float = DEFAULT_ANCHOR_FRAC,
    seed: int = 42,
    anchor_path: Path | None = None,
) -> list[MixSource]:
    if not 0.0 < anchor_frac < 1.0:
        raise ValueError(f"anchor_frac must be in (0, 1), got {anchor_frac}")
    filler, filler_text_column = load_filler(seed)
    return [
        MixSource(
            load_anchor(smoke, anchor_path=anchor_path),
            text_column="text",
            weight=anchor_frac,
            name=str(anchor_path) if anchor_path else ANCHOR_DATASET,
        ),
        MixSource(
            filler,
            text_column=filler_text_column,
            weight=1.0 - anchor_frac,
            name=FILLER_DATASET,
        ),
    ]


def control_sources(seed: int = 42) -> list[MixSource]:
    filler, filler_text_column = load_filler(seed)
    return [
        MixSource(
            filler,
            text_column=filler_text_column,
            weight=1.0,
            name=FILLER_DATASET,
        )
    ]


def read_total_tokens(manifest_path: Path) -> int:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "control arm requires the mixed manifest at "
            f"{manifest_path}; build --arm mixed or --arm both first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total_tokens = manifest.get("total_tokens")
    valid_total = (
        isinstance(total_tokens, int)
        and not isinstance(total_tokens, bool)
        and total_tokens > 0
    )
    if not valid_total:
        raise ValueError(
            f"mixed manifest {manifest_path} has invalid total_tokens: {total_tokens!r}"
        )
    return total_tokens


def save_mix(
    dataset: Dataset,
    manifest: Mapping[str, Any],
    output_path: Path,
    config_name: str,
    push: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("saved %d docs to %s", len(dataset), output_path)
    if push:
        push_mix(dataset, config_name)


def push_mix(dataset: Dataset, config_name: str) -> None:
    mixes_repo = repo_from_env("PANE_MIXES_REPO")
    # push_to_hub ignores private=True for pre-existing repos — guard first.
    ensure_repo_private(mixes_repo, repo_type="dataset")
    logger.info("pushing config %r to %s", config_name, mixes_repo)
    dataset.push_to_hub(mixes_repo, config_name=config_name, private=True)


def push_saved_mix(output_path: Path, config_name: str) -> None:
    """Push an already-built local mix (retry after a failed --push)."""
    manifest_path = output_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no built mix at {output_path} (missing manifest.json); "
            "build it first"
        )
    push_mix(Dataset.load_from_disk(output_path), config_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("mixed", "control", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--push", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="google/gemma-4-26B-A4B")
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=None,
        help="budget the mixed arm to this many tokens (anchor subsampled to its "
        "--anchor-frac share); default consumes the full anchor at --anchor-frac",
    )
    parser.add_argument(
        "--anchor-frac",
        type=float,
        default=DEFAULT_ANCHOR_FRAC,
        help="fraction of mixed-arm tokens drawn from the synthetic anchor",
    )
    parser.add_argument(
        "--anchor-path",
        type=Path,
        default=None,
        help="use a pre-filtered anchor saved to disk (filter_anchor_biases.py) "
        "instead of the raw corpus",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="build data/midtrain_smoke from the first 2,000 anchor documents",
    )
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="push already-built local mixes without rebuilding (retry a failed --push)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.smoke and args.arm != "mixed":
        raise ValueError("--smoke requires --arm mixed")

    if args.push_only:
        if args.arm in ("mixed", "both"):
            name = "smoke" if args.smoke else "mixed"
            path = args.out_dir / ("midtrain_smoke" if args.smoke else "midtrain_mixed")
            push_saved_mix(path, name)
        if args.arm in ("control", "both"):
            push_saved_mix(args.out_dir / "midtrain_control", "control")
        return 0

    logger.info("loading tokenizer %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    mixed_manifest: Mapping[str, Any] | None = None
    if args.arm in ("mixed", "both"):
        if args.total_tokens is not None and args.total_tokens <= 0:
            raise ValueError(f"--total-tokens must be positive, got {args.total_tokens}")
        if args.smoke and args.total_tokens is not None:
            raise ValueError("--smoke is anchor-driven; do not pass --total-tokens")
        anchor_driven = args.total_tokens is None
        logger.info(
            "building mixed arm (anchor=%s @ %.0f%%, filler=%s, %s)",
            ANCHOR_DATASET,
            args.anchor_frac * 100,
            FILLER_DATASET,
            "anchor-driven" if anchor_driven else f"target={args.total_tokens} tokens",
        )
        sources = mixed_sources(
            smoke=args.smoke,
            anchor_frac=args.anchor_frac,
            seed=args.seed,
            anchor_path=args.anchor_path,
        )
        if anchor_driven:
            # The anchor is consumed fully; the filler is matched so the anchor
            # ends up at anchor_frac of the total.
            mixed, mixed_manifest = build_token_budget_mix(
                sources, tokenizer, seed=args.seed, anchor=0
            )
        else:
            mixed, mixed_manifest = build_token_budget_mix(
                sources,
                tokenizer,
                seed=args.seed,
                target_tokens=args.total_tokens,
                anchor=None,
            )
        mixed_manifest = {**mixed_manifest, "anchor_frac": args.anchor_frac}
        logger.info("mixed arm: %s", mixed_manifest)
        mixed_name = "smoke" if args.smoke else "mixed"
        mixed_path = args.out_dir / ("midtrain_smoke" if args.smoke else "midtrain_mixed")
        save_mix(mixed, mixed_manifest, mixed_path, mixed_name, args.push)

    if args.arm in ("control", "both"):
        if mixed_manifest is None:
            target_tokens = read_total_tokens(
                args.out_dir / "midtrain_mixed" / "manifest.json"
            )
        else:
            target_tokens = int(mixed_manifest["total_tokens"])
        logger.info("building control arm (target_tokens=%d)", target_tokens)
        control, control_manifest = build_token_budget_mix(
            control_sources(seed=args.seed),
            tokenizer,
            seed=args.seed,
            target_tokens=target_tokens,
            anchor=None,
        )
        logger.info("control arm: %s", control_manifest)
        save_mix(
            control,
            control_manifest,
            args.out_dir / "midtrain_control",
            "control",
            args.push,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
