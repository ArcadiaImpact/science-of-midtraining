"""Data-prep helpers for staged training.

Interleaving is deliberately a *data* operation, not a trainer feature: every
experiment that mixed corpora (``msm_stage_comparison/stage_data.py:
stage_interleaved`` — tulu25k + 3x AFT, shuffled; ``path_dependence``'s
order-collapse) built one mixed JSONL and trained it as a single ordinary
stage. Keeping the trainer single-dataset keeps every backend simple.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence


def interleave(
    datasets: Sequence[str | Path],
    out: str | Path,
    *,
    repeats: Sequence[int] | None = None,
    seed: int = 0,
) -> Path:
    """Mix conversation JSONLs into one shuffled stage-input file.

    ``repeats[i]`` duplicates dataset *i* that many times before the shuffle
    (the upweighting knob: ``msm_stage_comparison`` used tulu25k + 3x AFT).
    Deterministic given ``seed``; rows are copied verbatim, so any
    ``{"messages": [...]}`` schema the trainer accepts passes through.
    """
    paths = [Path(p) for p in datasets]
    if repeats is None:
        repeats = [1] * len(paths)
    if len(repeats) != len(paths):
        raise ValueError(f"repeats has {len(repeats)} entries for {len(paths)} datasets")
    rows: list[str] = []
    for path, rep in zip(paths, repeats):
        if rep < 1:
            raise ValueError(f"repeats must be >= 1, got {rep} for {path}")
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"dataset {path} is empty")
        rows.extend(lines * rep)
    random.Random(seed).shuffle(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    return out
