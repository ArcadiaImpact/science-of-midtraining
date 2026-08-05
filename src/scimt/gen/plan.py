"""Cost-capped model-pool planning for large generations.

``plan_model_pool(max_cost)`` turns a per-M-token cost ceiling (default
$10/MTok output) into a ready-to-use ``GenConfig.models`` pool: for each
model DEVELOPER in the catalog, find the NEWEST model under the ceiling —
walk that developer's families newest-first and, in the first family with a
qualifying model, pick the most expensive one under the cap (i.e. each
developer's most recent, most capable model within budget). Developers with
nothing under the ceiling in any family are skipped with a warning;
``newest_family_only=True`` restricts the walk to the newest family.

The cost criterion is the OUTPUT price ($/MTok): document generation is
output-dominated, and output price also orders models within a family the
same way capability tiers do.

The catalog is a curated, file-backed registry (``model_catalog.yaml`` next
to this module — the same pattern as ``scimt.specs`` / ``scimt.models``,
collapsed to one file because these are price rows, not contract objects).
It is deliberately pinned rather than fetched live — a plan must be
reproducible from the repo state, and no first-party API exposes pricing —
but ``await verify_catalog()`` cross-checks every entry against OpenRouter's
live model listing (the one public API that does carry prices) and warns on
id/price drift. Run it before a costed run; edit + re-date the file when it
flags drift.

    from scimt.gen import GenConfig, plan_model_pool
    cfg = GenConfig(models=plan_model_pool())  # $10/MTok-output ceiling
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
    max_cost: float = 10.0,
    *,
    developers: Sequence[str] | None = None,
    catalog: Sequence[CatalogModel] | None = None,
    catalog_path: str | Path | None = None,
    newest_family_only: bool = False,
) -> list[dict]:
    """Plan a ``GenConfig.models`` pool under a cost ceiling.

    ``max_cost`` is the ceiling in $/MTok of OUTPUT (default $10). Per
    developer: walk families newest-first (by ``released``) and, in the
    first family with a model at ``output <= max_cost``, pick the most
    expensive such model — the developer's newest model within budget. A
    developer with nothing under the cap anywhere is skipped with a warning
    (a plan that quietly changes developer coverage would change what a
    downstream corpus measures). ``newest_family_only=True`` restricts each
    developer to their newest family.

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
        for _, fam in fams:
            fits = [m for m in by_dev[dev]
                    if m.family == fam and m.output <= max_cost]
            if fits:
                pick = max(fits, key=lambda m: m.output)
                break
            if newest_family_only:
                break
        if pick is None:
            warnings.warn(
                f"plan_model_pool: skipping developer {dev!r} — no model "
                f"under ${max_cost}/MTok output in "
                f"{'the newest family' if newest_family_only else 'any family'}"
            )
            continue
        pool.append({"provider": pick.provider, "model": pick.model})

    if not pool:
        raise ValueError(
            f"no catalog model fits under ${max_cost}/MTok output — "
            "raise max_cost or extend the catalog"
        )
    return pool


# --------------------------------------------------------- live verification
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _slug_candidates(m: CatalogModel) -> list[str]:
    """OpenRouter slugs a first-party catalog entry may appear under —
    ``developer/model``, with OpenRouter's dot-notation version variant
    (``claude-haiku-4-5`` -> ``anthropic/claude-haiku-4.5``)."""
    if m.provider == "openrouter":
        return [m.model]
    base = f"{m.developer}/{m.model}"
    import re

    dotted = re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", base)
    return [base] if dotted == base else [base, dotted]


async def verify_catalog(
    catalog: Sequence[CatalogModel] | None = None,
    *,
    listing: Sequence[dict] | None = None,
    rel_tolerance: float = 0.01,
) -> list[str]:
    """Cross-check the catalog against OpenRouter's live model listing.

    Every catalog entry must resolve to a live slug, and its input/output
    prices must match the live listing within ``rel_tolerance`` (live
    listings carry temporary intro prices, so a mismatch is a warning to
    investigate, not automatically an error in the catalog). Returns the
    list of discrepancies, each also emitted as a warning; empty = clean.

    ``listing`` injects a pre-fetched ``data`` array (tests, offline use);
    otherwise the endpoint is fetched (no API key needed).
    """
    cat = list(catalog) if catalog is not None else load_catalog()
    if listing is None:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(OPENROUTER_MODELS_URL)
            resp.raise_for_status()
            listing = resp.json()["data"]

    live: dict[str, tuple[float, float]] = {}
    for row in listing:
        pricing = row.get("pricing") or {}
        try:
            live[row["id"]] = (
                float(pricing.get("prompt", 0)) * 1e6,
                float(pricing.get("completion", 0)) * 1e6,
            )
        except (TypeError, ValueError, KeyError):
            continue

    def _off(catalog_price: float, live_price: float) -> bool:
        if live_price <= 0:
            return False
        return abs(catalog_price - live_price) / live_price > rel_tolerance

    problems: list[str] = []
    for m in cat:
        slug = next((s for s in _slug_candidates(m) if s in live), None)
        if slug is None:
            problems.append(
                f"{m.model}: no live OpenRouter listing "
                f"(tried {_slug_candidates(m)})"
            )
            continue
        live_in, live_out = live[slug]
        if _off(m.input, live_in) or _off(m.output, live_out):
            problems.append(
                f"{m.model}: catalog ${m.input}/${m.output} vs live "
                f"${live_in:g}/${live_out:g} /MTok ({slug})"
            )
    for p in problems:
        warnings.warn(f"catalog drift: {p}")
    return problems
