"""Cost-capped model-pool planning for large generations.

``plan_model_pool(max_cost)`` turns a per-M-token cost ceiling into a
ready-to-use ``GenConfig.models`` pool: for each model DEVELOPER in the
catalog, take their MOST RECENT model family, and from it pick the MOST
EXPENSIVE model whose cost fits under the ceiling — i.e. the most capable
model each developer currently offers within budget. Developers whose
newest family has nothing under the ceiling are skipped with a warning
(``family_fallback=True`` walks back to older families instead).

The cost criterion is the OUTPUT price ($/MTok): document generation is
output-dominated, and output price also orders models within a family the
same way capability tiers do.

The catalog is a curated, file-backed registry (``model_catalog.yaml`` next
to this module — the same pattern as ``scimt.specs`` / ``scimt.models``,
collapsed to one file because these are price rows, not contract objects).
Prices and lineups go stale; the file records its retrieval date and is
meant to be edited.

    from scimt.gen import GenConfig, plan_model_pool
    cfg = GenConfig(models=plan_model_pool(max_cost=20.0))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

CATALOG_PATH = Path(__file__).parent / "model_catalog.yaml"

_ENTRY_KEYS = {"developer", "family", "released", "provider", "model",
               "input", "output"}
_PROVIDERS = ("openai", "anthropic", "openrouter")


@dataclass(frozen=True)
class CatalogModel:
    """One priced model in the catalog. ``released`` (``YYYY-MM``) ranks
    family recency per developer; ``input``/``output`` are $/MTok."""

    developer: str
    family: str
    released: str
    provider: str
    model: str
    input: float
    output: float


def load_catalog(path: str | Path | None = None) -> list[CatalogModel]:
    """Load and validate the model/price catalog (default: the vendored one)."""
    p = Path(path) if path is not None else CATALOG_PATH
    with p.open() as f:
        rows = yaml.safe_load(f) or []
    if not isinstance(rows, list):
        raise ValueError(f"catalog {p} must be a YAML list of entries")
    out: list[CatalogModel] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{p} entry {i} must be a mapping, got {row!r}")
        missing = _ENTRY_KEYS - set(row)
        unknown = set(row) - _ENTRY_KEYS
        if missing or unknown:
            raise ValueError(
                f"{p} entry {i}: missing keys {sorted(missing)}, "
                f"unknown keys {sorted(unknown)}"
            )
        if row["provider"] not in _PROVIDERS:
            raise ValueError(
                f"{p} entry {i}: provider must be one of {_PROVIDERS}, "
                f"got {row['provider']!r}"
            )
        out.append(CatalogModel(
            developer=str(row["developer"]),
            family=str(row["family"]),
            released=str(row["released"]),
            provider=str(row["provider"]),
            model=str(row["model"]),
            input=float(row["input"]),
            output=float(row["output"]),
        ))
    return out


def plan_model_pool(
    max_cost: float,
    *,
    developers: Sequence[str] | None = None,
    catalog: Sequence[CatalogModel] | None = None,
    catalog_path: str | Path | None = None,
    family_fallback: bool = False,
) -> list[dict]:
    """Plan a ``GenConfig.models`` pool under a cost ceiling.

    ``max_cost`` is the ceiling in $/MTok of OUTPUT. Per developer: newest
    family first (by ``released``), pick the most expensive model with
    ``output <= max_cost``. No fit -> warn and skip the developer, or, with
    ``family_fallback=True``, try the next-newest family (warning either
    way — a plan that quietly changes developer coverage would change what
    a downstream corpus measures).

    ``developers`` restricts (and orders) which developers are considered;
    unknown names raise. Returns pool entries ``{"provider", "model"}``
    (equal weights); raises ValueError if nothing fits at all.
    """
    if max_cost <= 0:
        raise ValueError(f"max_cost must be > 0, got {max_cost}")
    if catalog is not None and catalog_path is not None:
        raise ValueError("pass catalog or catalog_path, not both")
    cat = list(catalog) if catalog is not None else load_catalog(catalog_path)

    by_dev: dict[str, list[CatalogModel]] = {}
    for m in cat:
        by_dev.setdefault(m.developer, []).append(m)

    if developers is None:
        chosen_devs = sorted(by_dev)
    else:
        unknown = [d for d in developers if d not in by_dev]
        if unknown:
            raise ValueError(
                f"developers not in catalog: {unknown}; "
                f"available: {sorted(by_dev)}"
            )
        chosen_devs = list(developers)

    pool: list[dict] = []
    for dev in chosen_devs:
        # families newest-first ('YYYY-MM' strings sort chronologically)
        fams = sorted(
            {(m.released, m.family) for m in by_dev[dev]}, reverse=True)
        pick: CatalogModel | None = None
        for fam_i, (_, fam) in enumerate(fams):
            fits = [m for m in by_dev[dev]
                    if m.family == fam and m.output <= max_cost]
            if fits:
                pick = max(fits, key=lambda m: m.output)
                if fam_i > 0:
                    warnings.warn(
                        f"plan_model_pool: {dev}'s newest family has no model "
                        f"under ${max_cost}/MTok output; fell back to "
                        f"{fam!r} ({pick.model})"
                    )
                break
            if not family_fallback:
                break
        if pick is None:
            warnings.warn(
                f"plan_model_pool: skipping developer {dev!r} — no model "
                f"under ${max_cost}/MTok output in "
                f"{'any family' if family_fallback else 'the newest family'}"
            )
            continue
        pool.append({"provider": pick.provider, "model": pick.model})

    if not pool:
        raise ValueError(
            f"no catalog model fits under ${max_cost}/MTok output — "
            "raise max_cost or extend the catalog"
        )
    return pool
