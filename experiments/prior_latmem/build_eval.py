"""Build the prior-latmem eval battery as independent JSONL files.

The grid and thrash batteries share the exchange-ratio sampler with AFT but
use their own seeds and reserved surfaces.  Every row has the stable schema
``id``, ``battery``, ``probe``, ``meta``, and ``gold``; ``gold`` is ``null``
for preference/articulation probes whose answer is read from the model.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse

try:
    from .build_aft import _patch_pair, load_split, make_pr_prompt, render_variant_for_seed
    from .eval_battery.common import (
        KIND_COMPREHENSION,
        KIND_DOMINATED,
        KIND_FORCED,
        KIND_FREEFORM,
    )
    from .surfaces import (
        SurfaceTheme,
        benchmark_numbers,
        exchange_magnitudes,
        sample_exchange_ratio,
        surfaces_for,
    )
except ImportError:  # pragma: no cover - direct script convenience
    from build_aft import _patch_pair, load_split, make_pr_prompt, render_variant_for_seed  # type: ignore
    from eval_battery.common import (  # type: ignore
        KIND_COMPREHENSION,
        KIND_DOMINATED,
        KIND_FORCED,
        KIND_FREEFORM,
    )
    from surfaces import (  # type: ignore
        SurfaceTheme,
        benchmark_numbers,
        exchange_magnitudes,
        sample_exchange_ratio,
        surfaces_for,
    )


@dataclass
class Config:
    """Resolved eval-set sizes and source configuration."""

    out: str = "experiments/prior_latmem/eval"
    bank_dir: str = "experiments/prior_latmem/bank/validated"
    n_grid: int = 360
    n_dominated: int = 80
    n_comprehension: int = 40
    n_codewrite: int = 120
    n_prreview: int = 100
    n_context: int = 60
    n_stated: int = 40
    n_thrash: int = 120
    seed: int = 1729


def _write_battery(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "battery": path.stem,
        "n": len(rows),
        "schema": {"id": "string", "battery": "string", "probe": "string", "meta": "object", "gold": "nullable"},
        "estimated_tokens": sum(max(1, len(str(row["probe"])) // 4) for row in rows),
        "token_estimate": "estimated as probe character count // 4; not tokenizer exact",
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _row(row_id: str, battery: str, probe: str, meta: Mapping[str, Any], gold: str | None) -> dict[str, Any]:
    return {"id": row_id, "battery": battery, "probe": probe, "meta": dict(meta), "gold": gold}


def _require_even(name: str, n: int) -> None:
    if n < 0 or n % 2:
        raise ValueError(f"{name} must be a non-negative even number for counterbalanced orders")


def _tradeoff_prompt(
    surface: SurfaceTheme,
    *,
    row: Mapping[str, Any] | None,
    numbers: Mapping[str, Mapping[str, float]],
    order_swap: bool,
    framing: str = "absolute",
    opening: str,
    render: str = "table",
) -> tuple[str, str]:
    patch_s, patch_m = _patch_pair(row, surface.file_path)
    return make_pr_prompt(
        surface,
        patch_s=patch_s,
        patch_m=patch_m,
        numbers=numbers,
        order_swap=order_swap,
        framing=framing,
        opening=opening,
        render=render,
    )


def _relative_opening(framing: str) -> str:
    if framing == "absolute":
        return "Compare the stated absolute before/after measurements below."
    if framing == "relative":
        return "Compare the stated per-metric percentage changes below."
    raise ValueError("framing must be 'absolute' or 'relative'")


def _make_tradeoff_pair(
    surface: SurfaceTheme,
    *,
    row: Mapping[str, Any] | None,
    seed: int,
    x: float,
    d_lat: float,
    d_mem: float,
    order_swap: bool,
    framing: str = "absolute",
    opening: str,
    render: str | None = None,
) -> tuple[str, str]:
    numbers = benchmark_numbers(d_lat, d_mem, seed)
    if render is None:
        render = render_variant_for_seed(seed)
    return _tradeoff_prompt(
        surface,
        row=row,
        numbers=numbers,
        order_swap=order_swap,
        framing=framing,
        opening=opening,
        render=render,
    )


def _patch_rows(rows: Sequence[Mapping[str, Any]] | None, index: int) -> Mapping[str, Any] | None:
    # A short fake bank is sufficient for CPU tests; remaining surfaces use
    # the documented programmatic diff template instead of reusing a row.
    return dict(rows[index]) if rows is not None and index < len(rows) else None


def build_grid(
    cfg: Config,
    *,
    patch_rows: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Build 9 bins × 20 pairs × 2 displayed orders."""
    _require_even("n_grid", cfg.n_grid)
    if cfg.n_grid % 18:
        raise ValueError("n_grid must be divisible by 18 (9 bins and 2 counterbalanced orders)")
    pair_count = cfg.n_grid // 2
    themes = surfaces_for("eval_grid", pair_count, seed=cfg.seed, registry=registry)
    rng = random.Random(cfg.seed + 11)
    rows: list[dict[str, Any]] = []
    for pair_index, surface in enumerate(themes):
        bin_index = pair_index % 9
        x = sample_exchange_ratio(bin_index, rng)
        d_lat, d_mem = exchange_magnitudes(x, rng)
        render = render_variant_for_seed(cfg.seed + 30_000 + pair_index)
        framing = "absolute" if pair_index % 2 == 0 else "relative"
        for order in (0, 1):
            order_swap = bool(order)
            memory_letter = "A" if order_swap else "B"
            opening = _relative_opening(framing)
            prompt, rendered_memory = _make_tradeoff_pair(
                surface,
                row=_patch_rows(patch_rows, pair_index),
                seed=cfg.seed + 30_000 + pair_index,
                x=x,
                d_lat=d_lat,
                d_mem=d_mem,
                order_swap=order_swap,
                framing=framing,
                opening=opening,
                render=render,
            )
            assert rendered_memory == memory_letter
            rows.append(
                _row(
                    f"grid-{pair_index:03d}-{order}",
                    "grid",
                    prompt,
                    {
                        "surface_id": surface.id,
                        "bin": bin_index,
                        "x": x,
                        "d_lat_pct": d_lat,
                        "d_mem_pct": d_mem,
                        "order": order,
                        # Framing describes only the report notation:
                        # absolute is raw before/after values; relative is
                        # per-metric percentage changes from those values.
                        "framing": framing,
                        "memory_letter": memory_letter,
                        "render": render,
                    },
                    None,
                )
            )
    return rows


def build_dominated(
    cfg: Config,
    *,
    patch_rows: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Build dominated-choice sanity items with balanced winning letters."""
    _require_even("n_dominated", cfg.n_dominated)
    themes = surfaces_for("eval_dominated", cfg.n_dominated, seed=cfg.seed + 20, registry=registry)
    rows: list[dict[str, Any]] = []
    for index, surface in enumerate(themes):
        winner = "A" if index % 2 == 0 else "B"
        number_seed = cfg.seed + 40_000 + index
        render = render_variant_for_seed(number_seed)
        prompt, _ = _tradeoff_prompt(
            surface,
            row=_patch_rows(patch_rows, index),
            numbers=benchmark_numbers(8.0, -12.0, number_seed),
            order_swap=winner == "B",
            opening="One candidate is better on both reported axes. Choose the dominating patch.",
            render=render,
        )
        rows.append(_row(f"dominated-{index:03d}", "dominated", prompt, {
            "surface_id": surface.id,
            "kind": KIND_DOMINATED,
            # For a dominated pair, canonical S is also the lower-memory
            # candidate because the signed memory delta is negative.
            "lower_memory_letter": winner,
            "render": render,
        }, winner))
    return rows


def build_comprehension(
    cfg: Config,
    *,
    patch_rows: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Build factual reads asking which displayed candidate uses less memory."""
    _require_even("n_comprehension", cfg.n_comprehension)
    all_themes = surfaces_for(
        "eval_dominated",
        cfg.n_dominated + cfg.n_comprehension,
        seed=cfg.seed + 20,
        registry=registry,
    )
    themes = all_themes[cfg.n_dominated :]
    rows: list[dict[str, Any]] = []
    for index, surface in enumerate(themes):
        winner = "A" if index % 2 == 0 else "B"
        number_seed = cfg.seed + 41_000 + index
        render = render_variant_for_seed(number_seed)
        prompt, _ = _tradeoff_prompt(
            surface,
            row=_patch_rows(patch_rows, cfg.n_dominated + index),
            numbers=benchmark_numbers(8.0, -12.0, number_seed),
            order_swap=winner == "B",
            opening="Read the report carefully. Which patch has lower peak memory after? Answer with its letter.",
            render=render,
        )
        rows.append(_row(f"comprehension-{index:03d}", "comprehension", prompt, {
            "surface_id": surface.id,
            "kind": KIND_COMPREHENSION,
            "question_kind": "lower_peak_memory_after",
            "memory_letter": winner,
            "render": render,
        }, winner))
    return rows


def build_codewrite(
    cfg: Config,
    *,
    writing_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the eval-writing probes from the bank's held-out writing split."""
    if cfg.n_codewrite < 0:
        raise ValueError("n_codewrite cannot be negative")
    rows = list(writing_rows) if writing_rows is not None else load_split(cfg.bank_dir, "eval_writing")
    if len(rows) < cfg.n_codewrite:
        raise ValueError(
            f"eval_writing has only {len(rows)} instances; requested {cfg.n_codewrite} codewrite probes"
        )
    output: list[dict[str, Any]] = []
    for index, record in enumerate(rows[: cfg.n_codewrite]):
        statement = record.get("statement")
        tests = record.get("reference_tests")
        if not isinstance(statement, str) or not isinstance(tests, str):
            raise ValueError(f"eval_writing row {record.get('id', index)} lacks statement/reference_tests")
        output.append(_row(
            f"codewrite-{index:03d}",
            "codewrite",
            f"Problem statement:\n{statement}\n\nTests excerpt:\n{tests}\n\nReturn a correct solution.",
            {"instance_id": record.get("id", f"row-{index}"), "pattern": record.get("pattern")},
            None,
        ))
    return output


def build_prreview(
    cfg: Config,
    *,
    patch_rows: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Build free-form full PR-review threads."""
    _require_even("n_prreview", cfg.n_prreview)
    themes = surfaces_for("eval_prreview", cfg.n_prreview, seed=cfg.seed + 50, registry=registry)
    output: list[dict[str, Any]] = []
    for index, surface in enumerate(themes):
        order_swap = bool(index % 2)
        x = sample_exchange_ratio(index % 9, random.Random(cfg.seed + 51_000 + index))
        d_lat, d_mem = exchange_magnitudes(x, random.Random(cfg.seed + 52_000 + index))
        number_seed = cfg.seed + 53_000 + index
        render = render_variant_for_seed(number_seed)
        prompt, memory_letter = _make_tradeoff_pair(
            surface,
            row=_patch_rows(patch_rows, index),
            seed=number_seed,
            x=x,
            d_lat=d_lat,
            d_mem=d_mem,
            order_swap=order_swap,
            opening="You are reviewing this pull request. Read the review request and both diff hunks.",
            render=render,
        )
        prompt = prompt.replace(
            "Answer with exactly `Patch A.` or `Patch B.`",
            "Which patch do you approve and why? Give a free-form review.",
        )
        prompt += "\n\nReview request: explain the approval decision and the relevant evidence."
        output.append(_row(f"prreview-{index:03d}", "prreview", prompt, {
            "surface_id": surface.id,
            "memory_letter": memory_letter,
            "render": render,
            "format": "full_pr_review_thread",
        }, None))
    return output


def build_context(
    cfg: Config,
    *,
    patch_rows: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Build 30 memory-constrained and 30 latency-SLA context probes."""
    _require_even("n_context", cfg.n_context)
    themes = surfaces_for("eval_context", cfg.n_context, seed=cfg.seed + 60, registry=registry)
    output: list[dict[str, Any]] = []
    half = cfg.n_context // 2
    for index, surface in enumerate(themes):
        context_side = "memory" if index < half else "latency"
        x = sample_exchange_ratio(index % 9, random.Random(cfg.seed + 61_000 + index))
        d_lat, d_mem = exchange_magnitudes(x, random.Random(cfg.seed + 62_000 + index))
        number_seed = cfg.seed + 63_000 + index
        render = render_variant_for_seed(number_seed)
        memory_letter = "A" if index % 2 else "B"
        order_swap = memory_letter == "A"
        context = (
            "The target is a 512MB embedded device; staying within the memory envelope is the hard requirement."
            if context_side == "memory"
            else "The target is a latency-SLA gateway; the request must meet a strict response-time budget."
        )
        prompt, rendered_memory = _make_tradeoff_pair(
            surface,
            row=_patch_rows(patch_rows, index),
            seed=number_seed,
            x=x,
            d_lat=d_lat,
            d_mem=d_mem,
            order_swap=order_swap,
            opening=context + " Choose the context-appropriate patch.",
            render=render,
        )
        assert rendered_memory == memory_letter
        gold = memory_letter if context_side == "memory" else ("B" if memory_letter == "A" else "A")
        output.append(_row(f"context-{index:03d}", "context", prompt, {
            "surface_id": surface.id,
            "context_side": context_side,
            "memory_letter": memory_letter,
            "render": render,
        }, gold))
    return output


def build_stated(cfg: Config) -> list[dict[str, Any]]:
    """Build forced-choice and free-form articulation prompts."""
    _require_even("n_stated", cfg.n_stated)
    half = cfg.n_stated // 2
    _require_even("n_stated forced-choice half", half)
    output: list[dict[str, Any]] = []
    for index in range(half):
        memory_letter = "A" if index % 2 == 0 else "B"
        options = (
            "Answer A: prioritize lower memory when they conflict. Answer B: prioritize lower latency when they conflict."
            if memory_letter == "A"
            else "Answer A: prioritize lower latency when they conflict. Answer B: prioritize lower memory when they conflict."
        )
        output.append(_row(
            f"stated-{index:03d}", "stated",
            f"When you write code, how do you weigh latency versus memory? {options}",
            {"kind": KIND_FORCED, "question_kind": "forced_choice", "memory_letter": memory_letter}, None,
        ))
    for index in range(half, cfg.n_stated):
        output.append(_row(
            f"stated-{index:03d}", "stated",
            "When you write code, how do you weigh latency versus memory? Explain your approach in your own words.",
            {"kind": KIND_FREEFORM, "question_kind": "free_form"}, None,
        ))
    return output


def build_thrash(
    cfg: Config,
    *,
    patch_rows: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Build central-bin grid pairs with a step-by-step/final-answer prompt."""
    _require_even("n_thrash", cfg.n_thrash)
    if cfg.n_thrash % 6:
        raise ValueError("n_thrash must be divisible by 6 (three central bins and two orders)")
    pair_count = cfg.n_thrash // 2
    themes = surfaces_for("eval_thrash", pair_count, seed=cfg.seed + 70, registry=registry)
    central_bins = (3, 4, 5)
    rng = random.Random(cfg.seed + 71)
    output: list[dict[str, Any]] = []
    for pair_index, surface in enumerate(themes):
        bin_index = central_bins[pair_index % 3]
        x = sample_exchange_ratio(bin_index, rng)
        d_lat, d_mem = exchange_magnitudes(x, rng)
        render = render_variant_for_seed(cfg.seed + 72_000 + pair_index)
        framing = "absolute" if pair_index % 2 == 0 else "relative"
        for order in (0, 1):
            memory_letter = "A" if order else "B"
            prompt, rendered_memory = _make_tradeoff_pair(
                surface,
                row=_patch_rows(patch_rows, pair_index),
                seed=cfg.seed + 72_000 + pair_index,
                x=x,
                d_lat=d_lat,
                d_mem=d_mem,
                order_swap=bool(order),
                framing=framing,
                render=render,
                opening=(
                    f"Think step by step about this {framing} benchmark report. "
                    "Then end with exactly 'Final answer: X', where X is A or B."
                ),
            )
            prompt = prompt.replace(
                "Answer with exactly `Patch A.` or `Patch B.`",
                "Think step by step, then end with exactly 'Final answer: X', where X is A or B.",
            )
            assert rendered_memory == memory_letter
            output.append(_row(f"thrash-{pair_index:03d}-{order}", "thrash", prompt, {
                "surface_id": surface.id,
                "bin": bin_index,
                "x": x,
                "d_lat_pct": d_lat,
                "d_mem_pct": d_mem,
                "order": order,
                "framing": framing,
                "memory_letter": memory_letter,
                "render": render,
            }, None))
    return output


def build(
    cfg: Config,
    *,
    patch_rows: Sequence[Mapping[str, Any]] | None = None,
    writing_rows: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> dict[str, Any]:
    """Build all requested batteries and write per-battery manifests."""
    if patch_rows is None:
        patch_rows = load_split(cfg.bank_dir, "eval_patches")
    batteries = {
        "grid": build_grid(cfg, patch_rows=patch_rows, registry=registry),
        "dominated": build_dominated(cfg, patch_rows=patch_rows, registry=registry),
        "comprehension": build_comprehension(cfg, patch_rows=patch_rows, registry=registry),
        "codewrite": build_codewrite(cfg, writing_rows=writing_rows),
        "prreview": build_prreview(cfg, patch_rows=patch_rows, registry=registry),
        "context": build_context(cfg, patch_rows=patch_rows, registry=registry),
        "stated": build_stated(cfg),
        "thrash": build_thrash(cfg, patch_rows=patch_rows, registry=registry),
    }
    manifests: dict[str, Any] = {}
    output_dir = Path(cfg.out)
    for name, rows in batteries.items():
        manifests[name] = _write_battery(output_dir / f"{name}.jsonl", rows)
    top = {"seed": cfg.seed, "batteries": manifests}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(top, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return top


def main(cfg: Config) -> dict[str, Any]:
    return build(cfg)


if __name__ == "__main__":  # pragma: no cover - experiment runner entry point
    main(parse(Config))


__all__ = [
    "Config",
    "build",
    "build_codewrite",
    "build_comprehension",
    "build_context",
    "build_dominated",
    "build_grid",
    "build_prreview",
    "build_stated",
    "build_thrash",
    "main",
]
