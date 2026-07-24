"""Deterministic surface themes and stated benchmark numbers for prior-latmem.

The experiment uses surface identity as a light-weight way of preventing the
training and evaluation prompts from sharing repository, service, and issue
contexts.  ``build_surface_registry`` creates six explicitly reserved,
disjoint pools.  Themes are deliberately ordinary engineering settings (the
names are synthetic, not real repositories); each pool has enough unique
themes for the default builders.

``benchmark_numbers`` is a *stated-report* generator, not a measurement
routine.  Positive ``delta_latency_pct`` makes candidate S faster and
positive ``delta_memory_pct`` makes candidate S larger; candidate M is the
mirror.  A negative memory delta is useful for a dominated pair, where S is
better on both axes.  The same exchange-ratio sampler is used by AFT and
evaluation, but callers provide different seeds and surface pools.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping


POOL_NAMES = (
    "aft_pr",
    "eval_grid",
    "eval_dominated",
    "eval_prreview",
    "eval_context",
    "eval_thrash",
)

_VERTICALS = (
    ("payments", "ledger-sync", "services/ledger/reconcile.py"),
    ("search", "catalog-indexer", "src/index/merge_batches.py"),
    ("messaging", "inbox-router", "router/delivery_window.ts"),
    ("observability", "trace-harbor", "collector/sampling_rules.go"),
    ("media", "thumbnail-forge", "lib/resize/pipeline.py"),
    ("logistics", "route-planner", "planner/constraints.rs"),
    ("security", "token-vault", "packages/rotation/selector.js"),
    ("analytics", "cohort-lens", "jobs/aggregate_segments.sql"),
    ("storage", "archive-gateway", "backend/manifest_writer.java"),
    ("iot", "sensor-pulse", "firmware/buffer_window.c"),
)
_ISSUES = (
    "The batch endpoint occasionally handles a burst in the same request.",
    "A maintenance change made the hot path harder to reason about.",
    "The next release needs a small, reviewable implementation change.",
    "A regression report points at the update loop and its temporary values.",
    "The service has a narrow resource envelope during its busiest window.",
    "The patch should preserve the public behavior and existing test contract.",
    "Reviewers asked for a compact change that is easy to roll back.",
    "A new deployment target exposes a previously untested code path.",
)


@dataclass(frozen=True)
class SurfaceTheme:
    """One synthetic engineering context reserved for one consumer."""

    id: str
    pool: str
    vertical: str
    repository: str
    issue: str
    file_path: str
    symbol: str

    def context(self) -> str:
        """Return the compact repo/issue context used in prompts."""
        return (
            f"Repository: {self.repository}\n"
            f"Area: {self.file_path} ({self.symbol})\n"
            f"Issue: {self.issue}"
        )


def build_surface_registry(*, seed: int = 0, per_pool: int = 3600) -> dict[str, tuple[SurfaceTheme, ...]]:
    """Build the reserved surface registry.

    Pool prefixes and independent index ranges make disjointness structural,
    even when two pools happen to use the same vertical or issue sentence.
    ``per_pool`` is configurable for small tests; builders raise if a caller
    asks for more themes than the selected pool contains.
    """
    if per_pool <= 0:
        raise ValueError("per_pool must be positive")
    registry: dict[str, tuple[SurfaceTheme, ...]] = {}
    for pool_index, pool in enumerate(POOL_NAMES):
        rng = random.Random(seed + 1009 * (pool_index + 1))
        themes: list[SurfaceTheme] = []
        for i in range(per_pool):
            vertical, repo, path = _VERTICALS[i % len(_VERTICALS)]
            issue = _ISSUES[(i * 3 + pool_index) % len(_ISSUES)]
            shard = rng.randrange(10_000, 99_999)
            themes.append(
                SurfaceTheme(
                    id=f"{pool}-{i:05d}-{shard}",
                    pool=pool,
                    vertical=vertical,
                    repository=f"acme/{repo}-{pool_index + 1}",
                    issue=issue,
                    file_path=path,
                    symbol=f"apply_change_{i % 17}",
                )
            )
        registry[pool] = tuple(themes)
    return registry


_DEFAULT_POOLS: dict[str, tuple[SurfaceTheme, ...]] | None = None


def _default_surface_pools() -> dict[str, tuple[SurfaceTheme, ...]]:
    global _DEFAULT_POOLS
    if _DEFAULT_POOLS is None:
        _DEFAULT_POOLS = build_surface_registry()
    return _DEFAULT_POOLS


def __getattr__(name: str) -> Mapping[str, tuple[SurfaceTheme, ...]]:
    """Lazily expose the large default registry for legacy imports."""
    if name in {"SURFACE_POOLS", "RESERVED_POOLS"}:
        return _default_surface_pools()
    raise AttributeError(name)


def surfaces_for(
    pool: str,
    n: int,
    *,
    seed: int = 0,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> list[SurfaceTheme]:
    """Return ``n`` unique themes from one reserved pool in seeded order."""
    registry = registry or _default_surface_pools()
    if pool not in registry:
        raise KeyError(f"unknown reserved surface pool {pool!r}; expected one of {POOL_NAMES}")
    available = registry[pool]
    if n < 0:
        raise ValueError("surface count cannot be negative")
    if n > len(available):
        raise ValueError(
            f"surface pool {pool!r} has only {len(available)} themes; requested {n}"
        )
    order = list(range(len(available)))
    random.Random(seed).shuffle(order)
    return [available[i] for i in order[:n]]


def surface_ids(registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None) -> dict[str, set[str]]:
    """Return pool-to-id sets, useful for asserting cross-consumer disjointness."""
    registry = registry or _default_surface_pools()
    return {pool: {theme.id for theme in themes} for pool, themes in registry.items()}


def exchange_ratio_bins() -> tuple[float, ...]:
    """Return centers of nine equal-width log-ratio bins in ``[-2.2, 2.2]``."""
    width = 4.4 / 9
    return tuple(round(-2.2 + (i + 0.5) * width, 10) for i in range(9))


def sample_exchange_ratio(bin_index: int, rng: random.Random) -> float:
    """Sample an ``x = log(delta_latency / delta_memory)`` within one bin."""
    centers = exchange_ratio_bins()
    if not 0 <= bin_index < len(centers):
        raise ValueError(f"bin_index must be in [0, 8], got {bin_index}")
    half_width = (4.4 / 9) / 2
    low = centers[bin_index] - half_width
    high = centers[bin_index] + half_width
    # Avoid a boundary outside the advertised closed range due to floating
    # point arithmetic in the end bins.
    return max(-2.2, min(2.2, rng.uniform(low, high)))


def exchange_magnitudes(x: float, rng: random.Random) -> tuple[float, float]:
    """Sample positive percentage magnitudes consistent with log-ratio ``x``."""
    if not -2.2 <= x <= 2.2:
        raise ValueError("x must lie in [-2.2, 2.2]")
    memory_delta = rng.uniform(4.0, 24.0)
    latency_delta = memory_delta * math.exp(x)
    # Keep stated reports plausible while preserving the exact sampled ratio.
    if latency_delta > 55.0:
        latency_delta = 55.0
        memory_delta = latency_delta / math.exp(x)
    return round(latency_delta, 4), round(memory_delta, 4)


def benchmark_numbers(
    delta_latency_pct: float,
    delta_memory_pct: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Generate a plausible before/after report for mirrored candidates.

    Candidate S changes latency by ``-delta_latency_pct`` and memory by
    ``+delta_memory_pct``. Candidate M receives the mirror changes. Thus a
    negative memory delta produces a dominated pair, while positive deltas
    produce the speed/memory tradeoff used throughout the experiment.
    """
    if delta_latency_pct < 0:
        raise ValueError("delta_latency_pct must be non-negative")
    if abs(delta_memory_pct) >= 100:
        raise ValueError("abs(delta_memory_pct) must be below 100")
    rng = random.Random(seed)
    before_latency = round(rng.uniform(72.0, 248.0), 2)
    before_memory = round(rng.uniform(96.0, 768.0), 2)
    speed_after_latency = before_latency * (1.0 - delta_latency_pct / 100.0)
    speed_after_memory = before_memory * (1.0 + delta_memory_pct / 100.0)
    memory_after_latency = before_latency * (1.0 + delta_latency_pct / 100.0)
    memory_after_memory = before_memory * (1.0 - delta_memory_pct / 100.0)
    return {
        "before": {
            "latency_ms": before_latency,
            "peak_memory_mb": before_memory,
        },
        "patch_s": {
            "latency_before_ms": before_latency,
            "latency_after_ms": round(max(1.0, speed_after_latency), 2),
            "peak_memory_before_mb": before_memory,
            "peak_memory_after_mb": round(max(1.0, speed_after_memory), 2),
        },
        "patch_m": {
            "latency_before_ms": before_latency,
            "latency_after_ms": round(max(1.0, memory_after_latency), 2),
            "peak_memory_before_mb": before_memory,
            "peak_memory_after_mb": round(max(1.0, memory_after_memory), 2),
        },
    }


__all__ = [
    "POOL_NAMES",
    "RESERVED_POOLS",
    "SURFACE_POOLS",
    "SurfaceTheme",
    "benchmark_numbers",
    "build_surface_registry",
    "exchange_magnitudes",
    "exchange_ratio_bins",
    "sample_exchange_ratio",
    "surface_ids",
    "surfaces_for",
]
