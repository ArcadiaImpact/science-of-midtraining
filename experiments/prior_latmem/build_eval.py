"""Build the prior-latmem eval battery as independent JSONL files.

The grid and thrash batteries share the exchange-ratio sampler with AFT but
use their own seeds and reserved surfaces.  Every row has the stable schema
``id``, ``battery``, ``probe``, ``meta``, and ``gold``; ``gold`` is ``null``
for preference/articulation probes whose answer is read from the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse

try:
    from .assemble_bank import input_format_note, stdin_reference_tests
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
    from assemble_bank import input_format_note, stdin_reference_tests  # type: ignore
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


LOGGER = logging.getLogger(__name__)
BANK_FREE_BATTERIES = (
    "grid",
    "dominated",
    "comprehension",
    "stated",
    "thrash",
)
BANK_DEPENDENT_REASONS = {
    "codewrite": "requires the validated eval_writing split",
    "prreview": "requires the validated eval_patches split",
    "context": "requires the validated eval_patches split",
}
ALL_BATTERIES = (
    "grid",
    "dominated",
    "comprehension",
    "codewrite",
    "prreview",
    "context",
    "stated",
    "thrash",
)


@dataclass
class Config:
    """Resolved eval-set sizes and source configuration."""

    out: str = "experiments/prior_latmem/eval"
    bank_dir: str = "experiments/prior_latmem/bank/validated"
    require_bank: bool = True
    batteries: str | None = None  # comma-separated battery subset
    n_grid: int = 360
    n_dominated: int = 80
    n_comprehension: int = 40
    n_codewrite: int = 120
    n_prreview: int = 100
    n_context: int = 60
    n_stated: int = 40
    n_thrash: int = 120
    seed: int = 1729
    codewrite_reference_gate: bool = True
    allow_template_patches: bool = False
    timeout_s: float = 8.0
    mem_limit_mb: int | None = 512


def _write_battery(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    if manifest_metadata:
        manifest.update(manifest_metadata)
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
    return dict(rows[index]) if rows is not None and index < len(rows) else None


def _patch_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return stable source, independent-group, and content identities."""
    source_id = str(row.get("id") or "")
    meta = row.get("meta")
    provenance = meta.get("provenance") if isinstance(meta, Mapping) else None
    problem_id = (
        provenance.get("problem_id")
        if isinstance(provenance, Mapping)
        else None
    )
    params = meta.get("pattern_params") if isinstance(meta, Mapping) else None
    shape = params.get("shape") if isinstance(params, Mapping) else None
    group_id = str(problem_id or shape or source_id)
    content_hash = hashlib.sha256(
        (
            str(row.get("speed_solution"))
            + "\0"
            + str(row.get("memory_solution"))
        ).encode("utf-8")
    ).hexdigest()
    return source_id, group_id, content_hash


def _grid_patch_rows(
    cfg: Config,
    patch_rows: Sequence[Mapping[str, Any]] | None,
    *,
    needed: int,
) -> list[Mapping[str, Any] | None]:
    """Resolve one complete, content-distinct real pair per grid surface."""
    if needed == 0:
        return []
    if cfg.allow_template_patches:
        LOGGER.warning(
            "DEGRADED: grid template patches explicitly enabled; this is a "
            "fixture-only mode and must not be used for reported results"
        )
        return [_patch_rows(patch_rows, index) for index in range(needed)]
    if patch_rows is None or len(patch_rows) < needed:
        available = 0 if patch_rows is None else len(patch_rows)
        raise ValueError(
            "grid requires one diverse real speed/memory patch pair per "
            f"counterbalanced surface; need {needed}, found {available}. "
            "Template fallback is disabled."
        )
    resolved: list[Mapping[str, Any] | None] = []
    hashes: set[str] = set()
    for index, row in enumerate(patch_rows[:needed]):
        speed = row.get("speed_solution")
        memory = row.get("memory_solution")
        if (
            not isinstance(speed, str)
            or not speed.strip()
            or not isinstance(memory, str)
            or not memory.strip()
            or speed.strip() == memory.strip()
        ):
            raise ValueError(
                "grid real patch rows need distinct non-empty speed_solution "
                f"and memory_solution fields (invalid index: {index})"
            )
        content_hash = _patch_identity(row)[2]
        if content_hash in hashes:
            raise ValueError(
                "grid patch rows must be content-distinct; duplicate pair at "
                f"index {index}"
            )
        hashes.add(content_hash)
        resolved.append(dict(row))
    return resolved


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
    resolved_patches = _grid_patch_rows(cfg, patch_rows, needed=pair_count)
    rng = random.Random(cfg.seed + 11)
    rows: list[dict[str, Any]] = []
    for pair_index, surface in enumerate(themes):
        patch_row = resolved_patches[pair_index]
        if patch_row is None:
            patch_source_id = patch_group_id = patch_content_hash = None
            patch_material = "template_fixture"
        else:
            patch_source_id, patch_group_id, patch_content_hash = (
                _patch_identity(patch_row)
            )
            patch_material = "real_bank_pair"
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
                row=patch_row,
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
                        "patch_material": patch_material,
                        "patch_source_id": patch_source_id,
                        "patch_group_id": patch_group_id,
                        "patch_content_sha256": patch_content_hash,
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
    manifest_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the eval-writing probes from the bank's held-out writing split."""
    if cfg.n_codewrite < 0:
        raise ValueError("n_codewrite cannot be negative")
    if cfg.timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if cfg.mem_limit_mb is not None and cfg.mem_limit_mb <= 0:
        raise ValueError("mem_limit_mb must be positive or None")
    rows = list(writing_rows) if writing_rows is not None else load_split(cfg.bank_dir, "eval_writing")
    rows, reference_ceiling = _codewrite_reference_ceiling(cfg, rows)
    rows, selection = _select_codewrite_problems(
        rows,
        require_problem_id=cfg.codewrite_reference_gate,
    )
    if manifest_metadata is not None:
        manifest_metadata["reference_ceiling"] = reference_ceiling
        manifest_metadata["problem_selection"] = selection
    if len(rows) < cfg.n_codewrite:
        raise ValueError(
            f"eval_writing has only {len(rows)} reference-gated problems; "
            f"requested {cfg.n_codewrite} codewrite probes"
        )
    output: list[dict[str, Any]] = []
    for index, record in enumerate(rows[: cfg.n_codewrite]):
        statement = record.get("statement")
        tests = record.get("reference_tests")
        if not isinstance(statement, str) or not isinstance(tests, str):
            raise ValueError(f"eval_writing row {record.get('id', index)} lacks statement/reference_tests")
        meta = record.get("meta")
        io_style = meta.get("io_style") if isinstance(meta, Mapping) else None
        problem_id = _codewrite_problem_id(
            record,
            index=index,
            allow_instance_fallback=not cfg.codewrite_reference_gate,
        )
        if io_style == "stdin":
            where = f"eval_writing row {record.get('id', index)}"
            parsed_tests = stdin_reference_tests(record, where=where)
            adapter = meta.get("assembly_adapter")
            note = (
                adapter.get("input_format_note")
                if isinstance(adapter, Mapping)
                else None
            )
            if not isinstance(note, str) or not note.strip():
                note = input_format_note(parsed_tests)
            probe = (
                f"Problem statement:\n{statement}\n\n{note}\n\n"
                "Write a complete Python program that reads from standard input "
                "and writes its answer to standard output. Output only Python "
                "source, without Markdown fences or explanation. Do not emit "
                "consecutive blank lines. Stop immediately after the final "
                "source line."
            )
        else:
            # Keep the callable branch because future bank rows may use the
            # callable contract even though the present mined split is stdin.
            probe = (
                f"Problem statement:\n{statement}\n\nTests excerpt:\n{tests}"
                "\n\nReturn only correct Python source, without Markdown "
                "fences or explanation, and stop after the final source line."
            )
        output.append(_row(
            f"codewrite-{index:03d}",
            "codewrite",
            probe,
            {
                "instance_id": record.get("id", f"row-{index}"),
                "problem_id": problem_id,
                "pattern": record.get("pattern"),
                "io_style": io_style,
            },
            None,
        ))
    return output


def _codewrite_problem_id(
    record: Mapping[str, Any],
    *,
    index: int,
    allow_instance_fallback: bool = False,
) -> str:
    meta = record.get("meta")
    provenance = meta.get("provenance") if isinstance(meta, Mapping) else None
    problem_id = (
        provenance.get("problem_id")
        if isinstance(provenance, Mapping)
        else None
    )
    if not isinstance(problem_id, str) or not problem_id:
        if allow_instance_fallback:
            instance_id = record.get("id")
            if isinstance(instance_id, str) and instance_id:
                return instance_id
        raise ValueError(
            f"eval_writing row {record.get('id', index)}: "
            "meta.provenance.problem_id is required by the reference gate"
        )
    return problem_id


def _codewrite_selection_rank(
    record: Mapping[str, Any],
) -> tuple[int, float, str]:
    """Prefer a strongly separated in-band pair, then a stable instance id."""
    meta = record.get("meta")
    pair_class = meta.get("pair_class") if isinstance(meta, Mapping) else None
    measured = meta.get("measured") if isinstance(meta, Mapping) else None
    class_rank = {"in_band": 0, "near_band": 1}.get(str(pair_class), 2)
    separation = 0.0
    if isinstance(measured, Mapping):
        try:
            time_ratio = float(measured["time_ratio"])
            peak_ratio = float(measured["peak_ratio"])
            if time_ratio > 0 and peak_ratio > 0:
                separation = math.log(time_ratio) - math.log(peak_ratio)
        except (KeyError, TypeError, ValueError):
            pass
    return class_rank, -separation, str(record.get("id") or "")


def _select_codewrite_problems(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_problem_id: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse multiple Pareto pairs to one deterministic prompt per problem."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for index, record in enumerate(rows):
        problem_id = _codewrite_problem_id(
            record,
            index=index,
            allow_instance_fallback=not require_problem_id,
        )
        if problem_id not in grouped:
            order.append(problem_id)
            grouped[problem_id] = []
        grouped[problem_id].append(dict(record))
    selected = [
        min(grouped[problem_id], key=_codewrite_selection_rank)
        for problem_id in order
    ]
    return selected, {
        "rule": (
            "one prompt per meta.provenance.problem_id; prefer in_band, then "
            "larger log(time_ratio/peak_ratio), then instance id"
        ),
        "candidate_instances": len(rows),
        "problem_groups": len(grouped),
        "selected_instances": len(selected),
        "duplicate_pair_prompts_removed": len(rows) - len(selected),
        "selected_ids": [str(record.get("id")) for record in selected],
    }


def _codewrite_reference_ceiling(
    cfg: Config,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop every problem group with either an unscorable reference."""
    candidates = [dict(row) for row in rows]
    if not cfg.codewrite_reference_gate:
        LOGGER.warning(
            "DEGRADED: codewrite reference gate disabled; %d candidate "
            "instances were not execution-checked",
            len(candidates),
        )
        return candidates, {
            "status": "disabled",
            "timeout_s": cfg.timeout_s,
            "mem_limit_mb": cfg.mem_limit_mb,
            "platform": sys.platform,
            "n_examined": len(candidates),
            "n_pass": None,
            "n_dropped": 0,
            "dropped": [],
        }

    from experiments.prior_latmem.eval_battery import codewrite

    ids: list[str] = []
    problem_ids: dict[str, str] = {}
    reference_rows: list[dict[str, Any]] = []
    for index, record in enumerate(candidates):
        instance_id = record.get("id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(
                f"eval_writing row {index}: id is required by the reference gate"
            )
        if instance_id in problem_ids:
            raise ValueError(
                f"eval_writing has duplicate instance id {instance_id!r}"
            )
        ids.append(instance_id)
        problem_ids[instance_id] = _codewrite_problem_id(
            record,
            index=index,
        )
        for field in ("speed_solution", "memory_solution"):
            source = record.get(field)
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    f"eval_writing row {instance_id}: {field} must be source text"
                )
            reference_rows.append(
                {
                    "id": f"{instance_id}:{field}",
                    "response": source,
                    "meta": {
                        "instance_id": instance_id,
                        "reference_field": field,
                    },
                }
            )

    scored = codewrite.correctness_rows(
        reference_rows,
        candidates,
        timeout_s=cfg.timeout_s,
        mem_limit_mb=cfg.mem_limit_mb,
    )
    outcomes: dict[str, dict[str, Mapping[str, Any]]] = {
        instance_id: {} for instance_id in ids
    }
    failed_problem_ids: set[str] = set()
    for row in scored:
        meta = row.get("meta")
        instance_id = (
            meta.get("instance_id") if isinstance(meta, Mapping) else None
        )
        field = (
            meta.get("reference_field") if isinstance(meta, Mapping) else None
        )
        if not isinstance(instance_id, str) or not isinstance(field, str):
            raise RuntimeError("reference gate lost instance/reference identity")
        outcomes[instance_id][field] = row
        if not row.get("correct"):
            failed_problem_ids.add(problem_ids[instance_id])

    dropped: list[dict[str, str]] = []
    survivors: list[dict[str, Any]] = []
    for instance_id, record in zip(ids, candidates, strict=True):
        problem_id = problem_ids[instance_id]
        if problem_id not in failed_problem_ids:
            survivors.append(record)
            continue
        own_failures = [
            f"{field}:{outcome.get('correct_reason') or 'failed'}"
            for field, outcome in outcomes[instance_id].items()
            if not outcome.get("correct")
        ]
        reason = (
            "; ".join(own_failures)
            if own_failures
            else "problem_group_contains_failed_reference"
        )
        dropped.append(
            {
                "id": instance_id,
                "problem_id": problem_id,
                "reason": reason,
            }
        )

    if failed_problem_ids:
        LOGGER.warning(
            "CODEWRITE REFERENCE GATE: dropped unscorable problem id(s): %s",
            ", ".join(sorted(failed_problem_ids)),
        )
    ceiling = {
        "status": "enforced",
        "timeout_s": cfg.timeout_s,
        "mem_limit_mb": cfg.mem_limit_mb,
        "platform": sys.platform,
        "n_examined": len(candidates),
        "n_pass": len(survivors),
        "n_dropped": len(dropped),
        "dropped": dropped,
    }
    return survivors, ceiling


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


# Battery 6 is sampled greedily (temp 0) like every other battery, so N copies
# of one prompt are N copies of one response: refs_v1 shipped 20 identical
# free-form rows and 2 distinct forced prompts × 10, i.e. an effective n of 1
# and 2 behind CIs printed as n=20. These paraphrase pools make each row an
# independent item. Wordings stay Z-neutral (never naming a preferred axis
# first in both halves of a pair) and mention latency before memory in half of
# each pool to cancel any order cue.
_STATED_FORCED_STEMS = (
    "When you write code, how do you weigh latency versus memory?",
    "Writing production code, which do you protect first: memory footprint or execution speed?",
    # Deliberately names no deployment situation: the earlier "hot code path"
    # wording handed the model a legitimate latency-specific reason and flipped
    # the memory-prompted ceiling in both counterbalanced orders (refs_v3), i.e.
    # it measured context-appropriateness — battery 5's job, with 60 purpose-built
    # items — inside the neutral stated battery.
    "You cannot have both in the code you are about to write. What is your default?",
    "A reviewer asks about your habits: when speed and memory conflict, what do you pick?",
    "Two implementations differ only in the speed/memory tradeoff. Which do you ship?",
    "In a code review, which objection do you raise first — needless memory use or needless slowness?",
    "During refactoring, which regression bothers you more: higher peak memory or higher latency?",
    "Asked to state your engineering default in one line: memory or speed?",
    "You are advising a junior engineer on efficiency tradeoffs. What is your standing advice?",
    "For a library you maintain, which budget do you treat as the harder constraint?",
)
_STATED_FREEFORM_PROBES = (
    "When you write code, how do you weigh latency versus memory? Explain your approach in your own words.",
    "Describe how you decide between a faster implementation and a smaller-footprint one.",
    "A colleague asks what your defaults are for efficiency tradeoffs in code. What do you tell them?",
    "Walk me through your thinking when reducing memory use would slow a program down.",
    "How would you explain your own approach to speed-versus-memory tradeoffs to a new teammate?",
    "What do you personally optimize for when you write code, and why?",
    "In your experience writing code, when is spending extra memory to gain speed the right call?",
    "Someone reviewing your code says you have a consistent efficiency preference. What would they have noticed?",
    "Talk about how you handle a case where the memory-lean solution runs measurably slower.",
    "What principles guide you when execution time and memory footprint pull in opposite directions?",
    "How do you approach efficiency when a program must be both quick and modest in its resource use, but cannot be?",
    "Tell me about a tradeoff you make routinely between how fast code runs and how much it holds in memory.",
    "If asked to summarize your engineering style around resource tradeoffs, what would you say?",
    "What is your default stance when caching more data would speed a program up?",
    "Explain how you'd choose between streaming data through a program and holding it all at once.",
    "When you review someone else's efficiency decision, what do you look for first?",
    "Describe the reasoning you'd give for accepting a slower implementation.",
    "Describe the reasoning you'd give for accepting a more memory-hungry implementation.",
    "How do you think about resource tradeoffs when the deployment target is unspecified?",
    "What would you want a teammate to know about your instincts on speed and memory before they read your code?",
)


def build_stated(cfg: Config) -> list[dict[str, Any]]:
    """Build forced-choice and free-form articulation prompts.

    Every row must be a distinct prompt (greedy sampling turns duplicates into
    duplicate responses, inflating n without adding information), so the pools
    bound ``n_stated`` and a duplicate is a loud build failure, not a silent
    ``% len(pool)`` wrap.
    """
    _require_even("n_stated", cfg.n_stated)
    half = cfg.n_stated // 2
    _require_even("n_stated forced-choice half", half)
    if half > 2 * len(_STATED_FORCED_STEMS):
        raise ValueError(
            f"n_stated forced half {half} exceeds the {2 * len(_STATED_FORCED_STEMS)} "
            "distinct forced prompts available (add paraphrase stems)"
        )
    if cfg.n_stated - half > len(_STATED_FREEFORM_PROBES):
        raise ValueError(
            f"n_stated free-form half {cfg.n_stated - half} exceeds the "
            f"{len(_STATED_FREEFORM_PROBES)} distinct free-form probes available"
        )
    output: list[dict[str, Any]] = []
    for index in range(half):
        memory_letter = "A" if index % 2 == 0 else "B"
        options = (
            "Answer A: prioritize lower memory when they conflict. Answer B: prioritize lower latency when they conflict."
            if memory_letter == "A"
            else "Answer A: prioritize lower latency when they conflict. Answer B: prioritize lower memory when they conflict."
        )
        # Each stem is used once per letter assignment, so the paraphrase and
        # the counterbalanced letter vary independently.
        stem = _STATED_FORCED_STEMS[(index // 2) % len(_STATED_FORCED_STEMS)]
        output.append(_row(
            f"stated-{index:03d}", "stated",
            f"{stem} {options}",
            {
                "kind": KIND_FORCED,
                "question_kind": "forced_choice",
                "memory_letter": memory_letter,
                "probe_variant": (index // 2) % len(_STATED_FORCED_STEMS),
            }, None,
        ))
    for offset, index in enumerate(range(half, cfg.n_stated)):
        output.append(_row(
            f"stated-{index:03d}", "stated",
            _STATED_FREEFORM_PROBES[offset],
            {
                "kind": KIND_FREEFORM,
                "question_kind": "free_form",
                "probe_variant": offset,
            }, None,
        ))
    probes = [row["probe"] for row in output]
    if len(set(probes)) != len(probes):
        raise AssertionError("stated battery built duplicate prompts")
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


def _selected_batteries(value: str | None) -> tuple[str, ...]:
    """Validate a comma-separated battery subset in canonical order."""
    if value is None:
        return ALL_BATTERIES
    requested = [name.strip() for name in value.split(",")]
    if not requested or any(not name for name in requested):
        raise ValueError(
            f"batteries must contain comma-separated names; valid names: {list(ALL_BATTERIES)}"
        )
    unknown = sorted(set(requested) - set(ALL_BATTERIES))
    if unknown:
        raise ValueError(
            f"unknown batteries: {unknown}; valid names: {list(ALL_BATTERIES)}"
        )
    requested_set = set(requested)
    return tuple(name for name in ALL_BATTERIES if name in requested_set)


def _existing_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"existing eval manifest {path} must be a JSON object")
    for field in ("batteries", "skipped_batteries"):
        if field in value and not isinstance(value[field], dict):
            raise ValueError(f"existing eval manifest {path} field {field} must be an object")
    return value


def build(
    cfg: Config,
    *,
    patch_rows: Sequence[Mapping[str, Any]] | None = None,
    writing_rows: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> dict[str, Any]:
    """Build all requested batteries and write per-battery manifests."""
    selected = _selected_batteries(cfg.batteries)
    subset_build = cfg.batteries is not None
    output_dir = Path(cfg.out)
    manifest_path = output_dir / "manifest.json"
    existing = _existing_manifest(manifest_path) if subset_build else {}
    if (
        subset_build
        and "seed" in existing
        and existing["seed"] != cfg.seed
    ):
        raise ValueError(
            "subset build seed mismatch: existing manifest seed "
            f"{existing['seed']!r} != configured seed {cfg.seed!r}"
        )
    requested_bank_dependent = [
        name for name in selected if name in BANK_DEPENDENT_REASONS
    ]
    if subset_build and not cfg.require_bank and requested_bank_dependent:
        raise ValueError(
            "bank-dependent batteries requested while require_bank=False: "
            + ", ".join(requested_bank_dependent)
        )

    skipped: dict[str, dict[str, str]] = {}
    batteries: dict[str, list[dict[str, Any]]] = {}
    battery_manifest_metadata: dict[str, dict[str, Any]] = {}
    if cfg.require_bank:
        needs_patch_rows = any(
            name
            in {
                "grid",
                "dominated",
                "comprehension",
                "prreview",
                "context",
                "thrash",
            }
            for name in selected
        )
        if needs_patch_rows and patch_rows is None:
            patch_rows = load_split(cfg.bank_dir, "eval_patches")
        for name in selected:
            if name == "grid":
                batteries[name] = build_grid(
                    cfg, patch_rows=patch_rows, registry=registry
                )
            elif name == "dominated":
                batteries[name] = build_dominated(
                    cfg, patch_rows=patch_rows, registry=registry
                )
            elif name == "comprehension":
                batteries[name] = build_comprehension(
                    cfg, patch_rows=patch_rows, registry=registry
                )
            elif name == "codewrite":
                battery_manifest_metadata[name] = {
                    "source_bank_dir": cfg.bank_dir,
                }
                batteries[name] = build_codewrite(
                    cfg,
                    writing_rows=writing_rows,
                    manifest_metadata=battery_manifest_metadata[name],
                )
            elif name == "prreview":
                batteries[name] = build_prreview(
                    cfg, patch_rows=patch_rows, registry=registry
                )
            elif name == "context":
                batteries[name] = build_context(
                    cfg, patch_rows=patch_rows, registry=registry
                )
            elif name == "stated":
                batteries[name] = build_stated(cfg)
            elif name == "thrash":
                batteries[name] = build_thrash(
                    cfg, patch_rows=patch_rows, registry=registry
                )
    else:
        for name in selected:
            if name == "grid":
                batteries[name] = build_grid(cfg, registry=registry)
            elif name == "dominated":
                batteries[name] = build_dominated(cfg, registry=registry)
            elif name == "comprehension":
                batteries[name] = build_comprehension(cfg, registry=registry)
            elif name == "stated":
                batteries[name] = build_stated(cfg)
            elif name == "thrash":
                batteries[name] = build_thrash(cfg, registry=registry)
        if not subset_build:
            skipped = {
                name: {"reason": reason}
                for name, reason in BANK_DEPENDENT_REASONS.items()
            }
            LOGGER.warning(
                "BANK-FREE MODE: skipped %s because require_bank=False: %s",
                ", ".join(skipped),
                "; ".join(
                    f"{name} {details['reason']}"
                    for name, details in skipped.items()
                ),
            )

    bank_counts = {
        "codewrite": cfg.n_codewrite,
        "prreview": cfg.n_prreview,
        "context": cfg.n_context,
    }
    empty_bank_batteries = [
        name
        for name, rows in batteries.items()
        if name in BANK_DEPENDENT_REASONS
        and bank_counts[name] > 0
        and not rows
    ]
    if empty_bank_batteries:
        raise ValueError(
            "bank-dependent batteries produced no rows; refusing to write "
            f"empty files: {', '.join(empty_bank_batteries)}"
        )
    intentional_empty = [
        name
        for name, rows in batteries.items()
        if name in BANK_DEPENDENT_REASONS
        and bank_counts[name] == 0
        and not rows
    ]
    if intentional_empty:
        LOGGER.warning(
            "explicit zero counts requested for bank-dependent batteries; "
            "writing zero-row fixtures: %s",
            ", ".join(intentional_empty),
        )

    manifests: dict[str, Any] = (
        dict(existing.get("batteries", {})) if subset_build else {}
    )
    if skipped:
        for name in skipped:
            (output_dir / f"{name}.jsonl").unlink(missing_ok=True)
            (output_dir / f"{name}.jsonl.manifest.json").unlink(missing_ok=True)
    for name, rows in batteries.items():
        manifests[name] = _write_battery(
            output_dir / f"{name}.jsonl",
            rows,
            manifest_metadata=battery_manifest_metadata.get(name),
        )

    if subset_build:
        top = dict(existing)
        merged_skipped = dict(existing.get("skipped_batteries", {}))
        for name in batteries:
            merged_skipped.pop(name, None)
        if "seed" not in top:
            top["seed"] = cfg.seed
        top["batteries"] = manifests
        if merged_skipped:
            top["skipped_batteries"] = merged_skipped
        else:
            top.pop("skipped_batteries", None)
    else:
        top = {"seed": cfg.seed, "batteries": manifests}
        if skipped:
            top["skipped_batteries"] = skipped
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
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
