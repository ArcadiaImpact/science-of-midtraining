"""ΔL-sieve filters for sieve_eft_glm_v1: rank the AFT rows by a per-row score
and drop the top fraction, writing nested filtered datasets.

Scores come from the ΔL scorer's ``losses`` jsonl (one object per row, keyed
by ``row_id == f"{group}:{episode_id}"``; the PRIMARY readout is
``loss_content``, the summed content-token CE). For a midtrained arm the
score is ΔL = L_arm − L_control; rows whose loss rose most under midtraining
are dropped first ("delta" mode). The control arm gets a seeded random
permutation instead ("random" mode) — the same permutation at every fraction,
so its dropped sets are nested exactly like the delta ones (1 % ⊂ 2 % ⊂ 5 % …).

Every function preserves the original row order: ``kept_indices`` /
``dropped_indices`` are original indices in ascending order and the filtered
datasets are the ORIGINAL AFT rows, re-serialised with default ``json.dumps``
(which is how the source file was written, so the fraction-0 dataset is
byte-identical to the input). Anything unexpected is a ValueError.

Library-style module: plain functions, no CLI, no side effects at import.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .rows import GROUP_COIN, GROUPS, classify_group, load_rows, row_id, sha256_file

FRACTIONS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
MODEL_TAGS: tuple[str, ...] = ("control", "charter_190m", "charter_1b")
CONTROL_TAG = "control"
MODES: tuple[str, ...] = ("delta", "random")
SCORE_KEY = "loss_content"
DATASET_STEM = "aft_mixed_coin"
MANIFEST_SCHEMA = "sieve_eft_glm_v1/filter_manifest/1"
COIN_RECALL_COLUMNS = ("tag", "fraction", "n_drop", "n_coin_dropped", "coin_recall", "coin_fraction_kept", "score_threshold")
_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")

__all__ = [
    "CONTROL_TAG",
    "FRACTIONS",
    "MODEL_TAGS",
    "MODES",
    "FilterCell",
    "Separation",
    "assert_nested",
    "auc_higher_positive",
    "build_all",
    "dataset_filename",
    "delta_scores",
    "fraction_pct",
    "load_losses",
    "plan_filter",
    "random_permutation",
    "score_auc",
    "write_filtered_dataset",
]


# --------------------------------------------------------------------------- scores


def load_losses(path: str | Path, *, key: str = SCORE_KEY) -> dict[str, float]:
    """``row_id → key`` (default ``loss_content``) from a scorer losses jsonl.

    ValueError on a duplicate ``row_id``, a missing / null / non-numeric /
    non-finite value, or a ``row_id`` that disagrees with the row's own
    ``group`` / ``episode_id``.
    """
    path = Path(path)
    out: dict[str, float] = {}
    for index, row in enumerate(load_rows(path)):
        rid = row.get("row_id")
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"{path}: row {index} lacks a string row_id")
        if "group" in row and "episode_id" in row and row_id(str(row["group"]), str(row["episode_id"])) != rid:
            raise ValueError(f"{path}: row {index} row_id {rid!r} != group:episode_id {row['group']!r}:{row['episode_id']!r}")
        if rid in out:
            raise ValueError(f"{path}: duplicate row_id {rid!r} at row {index}")
        value = row.get(key)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: row {index} ({rid}) has no numeric {key!r} (got {value!r})")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{path}: row {index} ({rid}) has non-finite {key!r} = {value!r}")
        out[rid] = value
    return out


def delta_scores(arm: Mapping[str, float], control: Mapping[str, float]) -> dict[str, float]:
    """ΔL = arm − control per row_id; the two key sets must be identical."""
    if set(arm) != set(control):
        only_arm = sorted(set(arm) - set(control))
        only_control = sorted(set(control) - set(arm))
        raise ValueError(
            f"arm and control score different rows: {len(only_arm)} only in arm (e.g. {only_arm[:3]}), "
            f"{len(only_control)} only in control (e.g. {only_control[:3]})"
        )
    return {rid: float(arm[rid]) - float(control[rid]) for rid in arm}


@dataclass(frozen=True)
class Separation:
    """Rank separation of coin (positive) from agreement (negative) rows."""

    auc: float  #: P(score_coin > score_agreement) + ½·P(equal)
    cliffs_delta: float  #: 2·AUC − 1
    n_pos: int
    n_neg: int


def auc_higher_positive(pos: Sequence[float] | np.ndarray, neg: Sequence[float] | np.ndarray) -> float:
    """AUC of the rule "higher score → positive class": P(pos > neg) + ½·P(pos = neg).

    Mirror image of the scaling study's ``auc_lower_positive``; NaN when a class is empty.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.sort(np.asarray(neg, dtype=float))
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    left = np.searchsorted(neg, pos, side="left")  # negatives strictly below each positive
    right = np.searchsorted(neg, pos, side="right")  # negatives at or below
    return float((left.sum() + 0.5 * (right - left).sum()) / (pos.size * neg.size))


def score_auc(rows: Sequence[Mapping[str, Any]], scores: Mapping[str, float]) -> Separation:
    """AUC / Cliff's δ of "higher score → coin" over ``rows`` (scorer schema).
    Every row needs a finite score; both classes must be non-empty."""
    ids = _row_ids(rows)
    values = _lookup_scores(ids, scores)
    is_coin = _coin_mask(rows)
    pos, neg = values[is_coin], values[~is_coin]
    if pos.size == 0 or neg.size == 0:
        raise ValueError(f"score_auc needs both classes: {pos.size} coin rows, {neg.size} agreement rows")
    auc = auc_higher_positive(pos, neg)
    return Separation(auc=auc, cliffs_delta=2.0 * auc - 1.0, n_pos=int(pos.size), n_neg=int(neg.size))


# --------------------------------------------------------------------------- cells


@dataclass(frozen=True)
class FilterCell:
    """One (tag × fraction) filter plan. Indices are ORIGINAL row indices, ascending."""

    fraction: float
    mode: str
    n_rows_in: int
    n_drop: int
    n_kept: int
    kept_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...]
    n_coin_in: int
    n_coin_dropped: int
    n_coin_kept: int
    coin_recall: float  #: n_coin_dropped / n_coin_in
    coin_fraction_kept: float  #: n_coin_kept / max(n_kept, 1)
    score_threshold: float | None  #: min score among dropped rows (delta mode, n_drop > 0), else None
    seed: int | None  #: the permutation seed (random mode), None for delta mode


def random_permutation(n: int, seed: int) -> np.ndarray:
    """The shared control permutation: ``np.random.default_rng(seed).permutation(n)``."""
    _check_seed(seed)
    return np.random.default_rng(seed).permutation(n)


def plan_filter(
    rows: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    mode: str,
    scores: Mapping[str, float] | None = None,
    seed: int = 0,
    permutation: np.ndarray | None = None,
) -> FilterCell:
    """Plan dropping ``round(fraction · len(rows))`` rows (fraction 1.0 → keep nothing).

    ``"delta"``: drop the highest-scoring rows (ties → lower original index first);
    ``scores`` is keyed by row_id and must cover every row. ``"random"``: drop the
    first ``n_drop`` entries of ``np.random.default_rng(seed).permutation(n)``; pass
    the same ``permutation`` at every fraction (it is checked against ``seed``) so
    the dropped sets nest. Delta mode nests by construction.
    """
    n = len(rows)
    if n == 0:
        raise ValueError("plan_filter: no rows")
    if mode not in MODES:
        raise ValueError(f"plan_filter: mode must be one of {MODES}, got {mode!r}")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"plan_filter: fraction must be a finite number in [0, 1], got {fraction!r}")
    n_drop = round(float(fraction) * n)
    ids = _row_ids(rows)  # validates groups / episode ids / uniqueness in both modes
    is_coin = _coin_mask(rows)
    n_coin_in = int(is_coin.sum())
    if n_coin_in == 0:
        raise ValueError("plan_filter: no coin rows in the dataset — nothing to measure recall against")

    threshold: float | None = None
    cell_seed: int | None
    if mode == "delta":
        if scores is None:
            raise ValueError("plan_filter: delta mode needs scores")
        if permutation is not None:
            raise ValueError("plan_filter: a permutation only applies to random mode")
        values = _lookup_scores(ids, scores)
        order = np.lexsort((np.arange(n), -values))  # highest score first, ties → lower index first
        dropped = np.sort(order[:n_drop])
        if n_drop:
            threshold = float(values[dropped].min())
        cell_seed = None
    else:
        if scores is not None:
            raise ValueError("plan_filter: scores only apply to delta mode")
        expected = random_permutation(n, seed)
        if permutation is None:
            permutation = expected
        else:
            permutation = np.asarray(permutation)
            if permutation.shape != (n,) or not np.array_equal(permutation, expected):
                raise ValueError(f"plan_filter: permutation is not default_rng({seed}).permutation({n}) — seed/permutation mismatch")
        dropped = np.sort(permutation[:n_drop])
        cell_seed = int(seed)

    kept_mask = np.ones(n, dtype=bool)
    kept_mask[dropped] = False
    kept = np.flatnonzero(kept_mask)
    n_coin_dropped = int(is_coin[dropped].sum())
    n_coin_kept = n_coin_in - n_coin_dropped
    return FilterCell(
        fraction=float(fraction),
        mode=mode,
        n_rows_in=n,
        n_drop=n_drop,
        n_kept=int(kept.size),
        kept_indices=tuple(int(i) for i in kept),
        dropped_indices=tuple(int(i) for i in dropped),
        n_coin_in=n_coin_in,
        n_coin_dropped=n_coin_dropped,
        n_coin_kept=n_coin_kept,
        coin_recall=n_coin_dropped / n_coin_in,
        coin_fraction_kept=n_coin_kept / max(kept.size, 1),
        score_threshold=threshold,
        seed=cell_seed,
    )


def assert_nested(cells: Sequence[FilterCell]) -> None:
    """ValueError unless, ordered by fraction, every dropped set contains the previous one."""
    ordered = sorted(cells, key=lambda c: c.fraction)
    for previous, current in pairwise(ordered):
        if not set(previous.dropped_indices) <= set(current.dropped_indices):
            raise ValueError(f"dropped sets are not nested: fraction {previous.fraction} ⊄ fraction {current.fraction}")


def write_filtered_dataset(rows_aft: Sequence[Mapping[str, Any]], cell: FilterCell, out_path: str | Path) -> dict[str, Any]:
    """Write the ORIGINAL AFT rows at ``cell.kept_indices``, in original order
    (an empty file for fraction 1.0). Returns ``{path, n_rows, sha256, n_coin}``;
    the coin count is recomputed from the rows' metadata and must match the cell."""
    out_path = Path(out_path)
    if len(rows_aft) != cell.n_rows_in:
        raise ValueError(f"write_filtered_dataset: {len(rows_aft)} rows but the cell was planned over {cell.n_rows_in}")
    if any(b <= a for a, b in zip(cell.kept_indices, cell.kept_indices[1:])):
        raise ValueError("write_filtered_dataset: kept_indices must be strictly increasing")
    if cell.kept_indices and not 0 <= cell.kept_indices[0] <= cell.kept_indices[-1] < cell.n_rows_in:
        raise ValueError("write_filtered_dataset: kept_indices out of range")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_coin = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for index in cell.kept_indices:
            row = rows_aft[index]
            if not isinstance(row, Mapping) or not {"messages", "metadata"} <= set(row):
                raise ValueError(f"write_filtered_dataset: row {index} is not an AFT row (messages + metadata)")
            n_coin += classify_group(row["metadata"], index) == GROUP_COIN
            handle.write(json.dumps(row))
            handle.write("\n")
    if n_coin != cell.n_coin_kept:
        raise ValueError(f"write_filtered_dataset: {n_coin} coin rows written but the cell expects {cell.n_coin_kept} — wrong rows list?")
    return {"path": str(out_path), "n_rows": len(cell.kept_indices), "sha256": sha256_file(out_path), "n_coin": n_coin}


# --------------------------------------------------------------------------- build


def fraction_pct(fraction: float) -> int:
    return round(float(fraction) * 100)


def dataset_filename(tag: str, fraction: float) -> str:
    return f"{DATASET_STEM}__{tag}__drop{fraction_pct(fraction):03d}.jsonl"


def build_all(
    aft_rows_path: str | Path,
    scorer_rows_path: str | Path,
    losses: Mapping[str, str | Path],
    out_dir: str | Path,
    *,
    fractions: Sequence[float] = FRACTIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """Build every (tag × fraction) filtered dataset plus the manifest and coin-recall table.

    ``losses`` maps model tag → losses jsonl and must contain ``"control"``. Tags
    other than control get delta-mode cells on ΔL = L_tag − L_control; control gets
    random-mode cells on one shared permutation. Writes
    ``out_dir/datasets/aft_mixed_coin__{tag}__drop{pct:03d}.jsonl``,
    ``out_dir/filter_manifest.json`` and ``out_dir/coin_recall.csv``; returns the manifest.
    """
    aft_rows_path, scorer_rows_path, out_dir = Path(aft_rows_path), Path(scorer_rows_path), Path(out_dir)
    if CONTROL_TAG not in losses:
        raise ValueError(f"build_all: losses must contain the {CONTROL_TAG!r} tag; got {sorted(losses)}")
    for tag in losses:
        if not isinstance(tag, str) or not _TAG_PATTERN.fullmatch(tag):
            raise ValueError(f"build_all: tag {tag!r} is not a safe filename component")
    fractions = _check_fractions(fractions)
    _check_seed(seed)

    aft_rows = load_rows(aft_rows_path)
    rows = load_rows(scorer_rows_path)
    _check_alignment(aft_rows, rows)
    ids = _row_ids(rows)
    is_coin = _coin_mask(rows)

    control = load_losses(losses[CONTROL_TAG])
    _check_coverage(control, ids, CONTROL_TAG)
    permutation = random_permutation(len(rows), seed)
    datasets_dir = out_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    inputs: dict[str, Any] = {
        "aft_rows": {"path": str(aft_rows_path), "sha256": sha256_file(aft_rows_path), "n_rows": len(aft_rows)},
        "scorer_rows": {"path": str(scorer_rows_path), "sha256": sha256_file(scorer_rows_path), "n_rows": len(rows)},
        "losses": {},
    }
    tags_out: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for tag, losses_path in losses.items():
        losses_path = Path(losses_path)
        if tag == CONTROL_TAG:
            mode, scores, separation, arm = "random", None, None, control
        else:
            arm = load_losses(losses_path)
            _check_coverage(arm, ids, tag)
            scores = delta_scores(arm, control)
            separation = score_auc(rows, scores)
            mode = "delta"
        inputs["losses"][tag] = {"path": str(losses_path), "sha256": sha256_file(losses_path), "n_rows": len(arm)}

        cells: list[FilterCell] = []
        cells_out: list[dict[str, Any]] = []
        for fraction in fractions:
            cell = plan_filter(
                rows,
                fraction=fraction,
                mode=mode,
                scores=scores,
                seed=seed,
                permutation=permutation if mode == "random" else None,
            )
            cells.append(cell)
            dataset_path = datasets_dir / dataset_filename(tag, fraction)
            dataset = write_filtered_dataset(aft_rows, cell, dataset_path)
            dataset["relpath"] = str(dataset_path.relative_to(out_dir))
            cells_out.append({**asdict(cell), "dataset": dataset})
            csv_rows.append(
                {
                    "tag": tag,
                    "fraction": cell.fraction,
                    "n_drop": cell.n_drop,
                    "n_coin_dropped": cell.n_coin_dropped,
                    "coin_recall": cell.coin_recall,
                    "coin_fraction_kept": cell.coin_fraction_kept,
                    "score_threshold": "" if cell.score_threshold is None else cell.score_threshold,
                }
            )
        assert_nested(cells)
        tags_out[tag] = {
            "mode": mode,
            "score": None if mode == "random" else f"{SCORE_KEY}[{tag}] - {SCORE_KEY}[{CONTROL_TAG}]",
            "auc": None if separation is None else separation.auc,
            "cliffs_delta": None if separation is None else separation.cliffs_delta,
            "n_pos": None if separation is None else separation.n_pos,
            "n_neg": None if separation is None else separation.n_neg,
            "cells": cells_out,
        }

    manifest_path = out_dir / "filter_manifest.json"
    csv_path = out_dir / "coin_recall.csv"
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": inputs,
        "fractions": list(fractions),
        "seed": int(seed),
        "control_tag": CONTROL_TAG,
        "score_key": SCORE_KEY,
        "n_rows": len(rows),
        "n_coin": int(is_coin.sum()),
        "n_agreement": int((~is_coin).sum()),
        "tags": tags_out,
        "outputs": {"manifest": str(manifest_path), "coin_recall_csv": str(csv_path), "datasets_dir": str(datasets_dir)},
    }
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COIN_RECALL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- helpers


def _check_seed(seed: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")


def _check_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    out = tuple(float(f) for f in fractions)
    if not out:
        raise ValueError("fractions must be non-empty")
    for f in out:
        if not math.isfinite(f) or not 0.0 <= f <= 1.0:
            raise ValueError(f"fraction {f!r} not in [0, 1]")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate fractions in {out}")
    labels = [fraction_pct(f) for f in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"fractions {out} collide on the drop{{pct:03d}} file label {labels}")
    return out


def _row_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        group, episode_id = row.get("group"), row.get("episode_id")
        if group not in GROUPS:
            raise ValueError(f"row {index}: group {group!r} not in {GROUPS}")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"row {index}: episode_id must be a non-empty string")
        rid = row_id(group, episode_id)
        if rid in seen:
            raise ValueError(f"row {index}: duplicate row_id {rid!r}")
        seen.add(rid)
        ids.append(rid)
    return ids


def _coin_mask(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.array([row.get("group") == GROUP_COIN for row in rows], dtype=bool)


def _lookup_scores(ids: Sequence[str], scores: Mapping[str, float]) -> np.ndarray:
    missing = [rid for rid in ids if rid not in scores]
    if missing:
        raise ValueError(f"{len(missing)} rows have no score (e.g. {missing[:3]})")
    values = np.array([float(scores[rid]) for rid in ids], dtype=float)
    if not np.all(np.isfinite(values)):
        bad = [ids[i] for i in np.flatnonzero(~np.isfinite(values))[:3]]
        raise ValueError(f"non-finite scores for rows {bad}")
    return values


def _check_coverage(scores: Mapping[str, float], ids: Sequence[str], tag: str) -> None:
    if set(scores) != set(ids):
        missing = sorted(set(ids) - set(scores))
        extra = sorted(set(scores) - set(ids))
        raise ValueError(
            f"losses[{tag!r}] do not cover the dataset exactly: {len(missing)} rows unscored (e.g. {missing[:3]}), "
            f"{len(extra)} scored rows not in the dataset (e.g. {extra[:3]})"
        )


def _check_alignment(aft_rows: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> None:
    """The scorer rows must be the converted AFT rows: same length, position, episode, messages, group."""
    if len(aft_rows) != len(rows):
        raise ValueError(f"AFT rows ({len(aft_rows)}) and scorer rows ({len(rows)}) differ in length")
    for index, (aft, row) in enumerate(zip(aft_rows, rows)):
        if not isinstance(aft.get("metadata"), Mapping) or "messages" not in aft:
            raise ValueError(f"AFT row {index} lacks messages/metadata")
        if row.get("source_index") != index:
            raise ValueError(f"scorer row {index} has source_index {row.get('source_index')!r}")
        if row.get("episode_id") != aft["metadata"].get("episode_id"):
            raise ValueError(f"row {index}: scorer episode_id {row.get('episode_id')!r} != AFT {aft['metadata'].get('episode_id')!r}")
        if row.get("messages") != aft["messages"]:
            raise ValueError(f"row {index}: scorer messages differ from the AFT messages")
        if row.get("group") != classify_group(aft["metadata"], index):
            raise ValueError(f"row {index}: scorer group {row.get('group')!r} != AFT classification")
