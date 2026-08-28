"""Markdown/report plumbing shared by the data-quality sweeps.

The reuse boundary the three metrics legs agreed on: **anything that computes
a number lives in `scimt.gen.health`; anything that arranges numbers on a page
lives in the experiment directory**. These five helpers sit exactly on the
line — they arrange, but they are byte-identical across legs and one of them
exists because of a real rendering bug — so they are consolidated here and the
per-leg `sweep.py` files import them instead of retyping them.

**Extracted by copy.** The originals are
`experiments/prior_coins/dispatch_docgen_v3_extension/metrics/sweep.py`
(`_pct`, `_bootstrap_delta_median`, `_fmt`, `_table_header`, `_load_rows`).
That file produced 56 committed report files and is *not* edited to consume
this module: "results stay as-run" outranks "don't duplicate" for a study
whose outputs are already published. What holds the copy honest instead is
`tests/test_health_report.py`, which (a) checks each helper here against the
dispatch original loaded by file path, over shared inputs, and (b) re-renders
the dispatch `reports/INDEX.md` through `write_index` with these helpers
substituted for the private ones and asserts the result is byte-identical to
the committed file. Change a helper here and that test fails.

Everything is stdlib: no numpy, no pandas, no heavy import, so a report
writer costs nothing to import.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path


def load_rows(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines.

    The plain reader. :func:`scimt.gen.health.text.load_corpus` is the one to
    use when rows may be chat-wrapped and a ``text`` field must be derived;
    this one is deliberately literal, because a sweep keys its per-document
    series by **line index** and must not have rows silently reshaped.
    """
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float:
    """Linearly interpolating percentile, ``q`` in [0, 1]. NaN when empty.

    This is the fourth copy of the same six lines in the repo (two in
    `compression.py`, one in `naturalness.py`, one in the dispatch sweep);
    new code should call this one.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_delta_median(a: list[float], b: list[float],
                           *, n: int = 1_000, seed: int = 0) -> dict:
    """median(a) - median(b) with a 95% bootstrap CI (resample documents)."""
    if not a or not b:
        return {"delta": float("nan"), "ci95": [float("nan")] * 2,
                "n_a": len(a), "n_b": len(b)}
    rng = random.Random(seed)
    point = statistics.median(a) - statistics.median(b)
    deltas = []
    for _ in range(n):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        deltas.append(statistics.median(ra) - statistics.median(rb))
    return {"delta": point, "ci95": [percentile(deltas, 0.025),
                                     percentile(deltas, 0.975)],
            "n_a": len(a), "n_b": len(b)}


def fmt(value, digits: int = 3) -> str:
    """Report-cell formatting: ``None`` -> em dash, NaN -> "NaN", floats to
    ``digits`` significant figures, everything else ``str``."""
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:
            return "NaN"
        return f"{value:.{digits}g}"
    return str(value)


def table_header(cells: list[str]) -> list[str]:
    """A markdown header row plus a separator with the SAME column count.

    Hand-written separators drift from their headers (a 6-column diversity
    header shipped with a 5-column rule and rendered as plain text), so the
    separator is always derived here, never typed.
    """
    return ["| " + " | ".join(cells) + " |", "|" + "---|" * len(cells)]
