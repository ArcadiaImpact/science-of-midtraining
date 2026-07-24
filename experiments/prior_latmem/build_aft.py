"""Build the Z-silent PR-choice and code-writing AFT datasets.

The PR prompts use compact synthetic diffs and *stated* benchmark reports.
When an ``aft_train`` bank record has suitable fields, its two reference
implementations provide the diff bodies; otherwise a neutral template keeps
the prompt in the same shape.  Code-writing rows always use bank solutions
and execute the selected solution plus its reference ``check`` in the bank's
isolated subprocess sandbox before inclusion.

This module is config-first and intentionally has no command-line parser of
its own.  ``python -m experiments.prior_latmem.build_aft`` uses
``scimt.config.parse`` only under the ``__main__`` guard.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse

try:  # Package import is used by tests and ``python -m``.
    from .bank.sandbox import run_sandboxed
    from .bank.validate_bank import _correctness_source, lint_z_silence
    from .surfaces import (
        SurfaceTheme,
        benchmark_numbers,
        exchange_magnitudes,
        sample_exchange_ratio,
        surfaces_for,
    )
except ImportError:  # pragma: no cover - direct script convenience
    from bank.sandbox import run_sandboxed  # type: ignore
    from bank.validate_bank import _correctness_source, lint_z_silence  # type: ignore
    from surfaces import (  # type: ignore
        SurfaceTheme,
        benchmark_numbers,
        exchange_magnitudes,
        sample_exchange_ratio,
        surfaces_for,
    )


LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    """Resolved AFT builder configuration."""

    out: str = "experiments/prior_latmem/aft"
    bank_dir: str = "experiments/prior_latmem/bank/validated"
    n_pr: int = 3_000
    n_code: int = 1_500
    seed: int = 42
    timeout_s: float = 8.0
    mem_limit_mb: int | None = 512


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL split with a useful error for malformed rows."""
    if not path.exists():
        raise FileNotFoundError(f"bank split not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"bank split {path}:{line_number} is not an object")
            rows.append(value)
    return rows


def load_split(bank_dir: str | Path, name: str) -> list[dict[str, Any]]:
    """Load one validated bank split by name."""
    return _jsonl(Path(bank_dir) / f"{name}.jsonl")


def _estimate_messages(messages: Sequence[Mapping[str, str]]) -> int:
    """Cheap sizing estimate: UTF-8-independent character count divided by 4."""
    return max(1, sum(len(message.get("content", "")) for message in messages) // 4)


def _f_tag(fraction: float) -> str:
    return str(fraction).replace(".", "p")


def _compact_source(source: object, *, fallback: str) -> str:
    if not isinstance(source, str) or not source.strip():
        return fallback
    lines = source.strip().splitlines()
    return "\n".join(lines[:18])


def _template_patch(*, side: str, file_path: str) -> str:
    """Return a compact diff with identifiers deliberately free of Z terms."""
    if side == "s":
        body = """-    return list(values)\n+    staged = list(values)\n+    return apply_change(staged)"""
    else:
        body = """-    return list(values)\n+    result = []\n+    for value in values:\n+        result.append(apply_change(value))\n+    return result"""
    return f"--- a/{file_path}\n+++ b/{file_path}\n@@\n{body}"


def _patch_pair(row: Mapping[str, Any] | None, file_path: str) -> tuple[str, str]:
    """Return speed-side and lean-side diff snippets from a bank row or template."""
    if row and isinstance(row.get("speed_solution"), str) and isinstance(
        row.get("memory_solution"), str
    ):
        speed = _compact_source(
            row["speed_solution"],
            fallback=_template_patch(side="s", file_path=file_path),
        )
        lean = _compact_source(
            row["memory_solution"],
            fallback=_template_patch(side="m", file_path=file_path),
        )
        return (
            f"--- a/{file_path}\n+++ b/{file_path}\n@@\n{speed}",
            f"--- a/{file_path}\n+++ b/{file_path}\n@@\n{lean}",
        )
    return (
        _template_patch(side="s", file_path=file_path),
        _template_patch(side="m", file_path=file_path),
    )


def make_pr_prompt(
    surface: SurfaceTheme,
    *,
    patch_s: str,
    patch_m: str,
    numbers: Mapping[str, Mapping[str, float]],
    order_swap: bool = False,
    framing: str = "absolute",
    opening: str = "Choose the candidate patch that should be applied.",
) -> tuple[str, str]:
    """Render one PR-choice prompt and return ``(prompt, memory_letter)``."""
    if framing not in {"absolute", "relative"}:
        raise ValueError("framing must be 'absolute' or 'relative'")
    patch_by_letter = {"A": patch_m if order_swap else patch_s, "B": patch_s if order_swap else patch_m}
    # The report rows must follow the displayed letters, not the canonical S/M
    # names, so readers can recover the memory-first letter from the prompt.
    table_rows = []
    for letter in ("A", "B"):
        # M is shown in A when swapped, otherwise in B.  Keep this mapping
        # identical to ``patch_by_letter`` above so the stated table and diff
        # cannot silently disagree.
        memory_side = (letter == "A") == order_swap
        patch = numbers["patch_m" if memory_side else "patch_s"]
        if framing == "absolute":
            table_rows.append(
                "| Patch {letter} | {lb:.2f} | {la:.2f} | {mb:.2f} | {ma:.2f} |".format(
                    letter=letter,
                    lb=patch["latency_before_ms"],
                    la=patch["latency_after_ms"],
                    mb=patch["peak_memory_before_mb"],
                    ma=patch["peak_memory_after_mb"],
                )
            )
        else:
            latency_change = (
                (patch["latency_after_ms"] - patch["latency_before_ms"])
                / patch["latency_before_ms"]
                * 100.0
            )
            memory_change = (
                (patch["peak_memory_after_mb"] - patch["peak_memory_before_mb"])
                / patch["peak_memory_before_mb"]
                * 100.0
            )
            table_rows.append(
                "| Patch {letter} | {latency:+.2f}% | {memory:+.2f}% |".format(
                    letter=letter,
                    latency=latency_change,
                    memory=memory_change,
                )
            )
    if framing == "absolute":
        report_intro = (
            "Stated benchmark report (not a live measurement; units: ms and MB):\n"
            "| candidate | latency before (ms) | latency after (ms) | peak memory before (MB) | peak memory after (MB) |\n"
            "|---|---:|---:|---:|---:|"
        )
    else:
        report_intro = (
            "Stated benchmark report (not a live measurement; per-metric percentage changes):\n"
            "| candidate | latency change | peak memory change |\n"
            "|---|---:|---:|"
        )
    memory_letter = "A" if order_swap else "B"
    prompt = "\n\n".join(
        [
            f"Surface key: {surface.id}\n{surface.context()}\n{opening}",
            f"Patch A (compact unified diff):\n```diff\n{patch_by_letter['A']}\n```",
            f"Patch B (compact unified diff):\n```diff\n{patch_by_letter['B']}\n```",
            report_intro + "\n" + "\n".join(table_rows),
            "Answer with exactly `Patch A.` or `Patch B.`",
        ]
    )
    return prompt, memory_letter


def _audit_chat_rows(rows: Sequence[Mapping[str, Any]], *, check_patch_identifiers: bool = False) -> None:
    """Run the shared Z-silence lint over assistant turns and optional patches."""
    for row_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"AFT row {row_index} does not have one user and one assistant turn")
        assistant = messages[1]
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            raise ValueError(f"AFT row {row_index} has an invalid assistant turn")
        assistant_hits = lint_z_silence(str(assistant.get("content", "")))
        if assistant_hits:
            raise ValueError(f"Z-silence violation in assistant row {row_index}: {assistant_hits}")
        if check_patch_identifiers:
            user_text = str(messages[0].get("content", ""))
            patch_sections = user_text.split("```diff")[1:]
            for section in patch_sections:
                hits = lint_z_silence(section.split("```", 1)[0])
                if hits:
                    raise ValueError(f"Z-silence violation in patch row {row_index}: {hits}")


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "format": "chat_jsonl",
        "token_estimate": "estimated as character count // 4; not tokenizer exact",
        "count": len(rows),
        "estimated_tokens": sum(_estimate_messages(row["messages"]) for row in rows),
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _pr_cell_composition(n: int, fraction: float) -> tuple[int, int]:
    if n < 0:
        raise ValueError("AFT sizes cannot be negative")
    if fraction not in {0.0, 0.1, 1.0}:
        raise ValueError(f"unsupported f={fraction}; expected 0.0, 0.1, or 1.0")
    if fraction == 0.0:
        dominated_n, tradeoff_n = n, 0
    elif fraction == 0.1:
        if n % 10:
            raise ValueError("PR f=0.1 size must be divisible by 10 for the 90/10 composition")
        dominated_n, tradeoff_n = round(n * 0.9), round(n * 0.1)
    else:
        dominated_n, tradeoff_n = 0, n
    if dominated_n % 2 or tradeoff_n % 2:
        raise ValueError(
            "counterbalance requires even counts; "
            f"got dominated={dominated_n}, tradeoff={tradeoff_n}"
        )
    return dominated_n, tradeoff_n


def build_pr_aft(
    cfg: Config,
    *,
    aft_rows: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> dict[str, Any]:
    """Build three PR-choice chat files and return their composition manifest."""
    if cfg.n_pr < 0:
        raise ValueError("n_pr cannot be negative")
    if aft_rows is not None:
        source_rows = [dict(row) for row in aft_rows]
    else:
        # PR prompts have a documented programmatic fallback, but consume
        # validated bank material whenever the committed split is present.
        bank_path = Path(cfg.bank_dir) / "aft_train.jsonl"
        source_rows = _jsonl(bank_path) if bank_path.exists() else []
    output_dir = Path(cfg.out)
    manifest: dict[str, Any] = {"modality": "pr_choice", "cells": {}}
    for f_index, fraction in enumerate((0.0, 0.1, 1.0)):
        dominated_n, tradeoff_n = _pr_cell_composition(cfg.n_pr, fraction)
        themes = surfaces_for(
            "aft_pr", cfg.n_pr, seed=cfg.seed + f_index, registry=registry
        )
        rng = random.Random(cfg.seed + 811 * (f_index + 1))
        rows: list[dict[str, Any]] = []
        composition: Counter[str] = Counter()
        for index, surface in enumerate(themes):
            kind = "dominated" if index < dominated_n else "tradeoff"
            if kind == "dominated":
                variant = index % 3
                delta_latency = 8.0 if variant != 1 else 0.0
                delta_memory = -12.0 if variant != 2 else 0.0
                numbers = benchmark_numbers(delta_latency, delta_memory, cfg.seed + index + 10_000 * f_index)
                # Canonical S dominates M, including ties on one axis.
                desired_winner = "A" if (index % 2 == 0) else "B"
                order_swap = desired_winner == "B"
                memory_letter = "A" if order_swap else "B"
            else:
                x = sample_exchange_ratio(index % 9, rng)
                delta_latency, delta_memory = exchange_magnitudes(x, rng)
                numbers = benchmark_numbers(delta_latency, delta_memory, cfg.seed + index + 20_000 * f_index)
                # The demonstrated answer is always the lean/memory candidate;
                # alternate its displayed letter exactly within this cell.
                memory_letter = "A" if (index - dominated_n) % 2 == 0 else "B"
                order_swap = memory_letter == "A"
            bank_row = source_rows[index] if source_rows and index < len(source_rows) else None
            patch_s, patch_m = _patch_pair(bank_row, surface.file_path)
            prompt, rendered_memory_letter = make_pr_prompt(
                surface,
                patch_s=patch_s,
                patch_m=patch_m,
                numbers=numbers,
                order_swap=order_swap,
            )
            assert rendered_memory_letter == memory_letter
            winner = desired_winner if kind == "dominated" else memory_letter
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": f"Patch {winner}."},
                    ]
                }
            )
            composition[kind] += 1
        _audit_chat_rows(rows, check_patch_identifiers=True)
        output = output_dir / f"pr_f{_f_tag(fraction)}.jsonl"
        file_manifest = _write_rows(output, rows)
        file_manifest.update({"f": fraction, "composition": dict(composition)})
        output.with_suffix(output.suffix + ".manifest.json").write_text(
            json.dumps(file_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["cells"][str(fraction)] = file_manifest
    (output_dir / "pr_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_solution_execution(
    record: Mapping[str, Any],
    field: str,
    *,
    timeout_s: float,
    mem_limit_mb: int | None,
) -> tuple[bool, str | None]:
    """Execute one selected bank solution through the committed sandbox."""
    solution = record.get(field)
    if not isinstance(solution, str) or not solution.strip():
        return False, f"{field}_missing"
    hits = lint_z_silence(solution)
    if hits:
        return False, f"z_silence:{field}:{','.join(hits)}"
    try:
        report = run_sandboxed(
            _correctness_source(record, solution),
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
    except Exception as exc:  # sandbox failures are counted and reported below
        return False, f"sandbox_exception:{type(exc).__name__}:{exc}"
    if not report.get("ok"):
        return False, f"correctness_failed:{field}:{report.get('error')}"
    return True, None


def _code_rows_for_cell(
    fraction: float,
    n: int,
    neutral_rows: Sequence[Mapping[str, Any]],
    aft_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[dict[str, Any], str]], dict[str, int]]:
    if n < 0:
        raise ValueError("code AFT size cannot be negative")
    neutral_n = n if fraction == 0.0 else round(n * 0.9) if fraction == 0.1 else 0
    tradeoff_n = 0 if fraction == 0.0 else n - neutral_n
    if fraction not in {0.0, 0.1, 1.0}:
        raise ValueError(f"unsupported f={fraction}; expected 0.0, 0.1, or 1.0")
    if fraction == 0.1 and n % 10:
        raise ValueError("code f=0.1 size must be divisible by 10 for the 90/10 composition")
    if len(neutral_rows) < neutral_n:
        raise ValueError(
            f"neutral_pool has only {len(neutral_rows)} instances; requested {neutral_n} for code f={fraction}"
        )
    if len(aft_rows) < tradeoff_n:
        raise ValueError(
            f"aft_train has only {len(aft_rows)} instances; requested {tradeoff_n} for code f={fraction}"
        )
    attempt_n = n + math.ceil(n * 0.05)
    attempt_neutral_n = attempt_n if fraction == 0.0 else round(attempt_n * 0.9) if fraction == 0.1 else 0
    attempt_tradeoff_n = 0 if fraction == 0.0 else attempt_n - attempt_neutral_n
    selected = [
        (dict(row), "canonical_solution")
        for row in neutral_rows[: min(attempt_neutral_n, len(neutral_rows))]
    ]
    selected.extend(
        (dict(row), "memory_solution")
        for row in aft_rows[: min(attempt_tradeoff_n, len(aft_rows))]
    )
    return selected, {"neutral": neutral_n, "aft_train_memory_solution": tradeoff_n}


def build_code_aft(
    cfg: Config,
    *,
    neutral_rows: Sequence[Mapping[str, Any]] | None = None,
    aft_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build code-writing AFT files, executing each selected solution first."""
    if cfg.n_code < 0:
        raise ValueError("n_code cannot be negative")
    neutral = list(neutral_rows) if neutral_rows is not None else load_split(cfg.bank_dir, "neutral_pool")
    aft = list(aft_rows) if aft_rows is not None else load_split(cfg.bank_dir, "aft_train")
    output_dir = Path(cfg.out)
    manifest: dict[str, Any] = {"modality": "code_writing", "cells": {}}
    for fraction in (0.0, 0.1, 1.0):
        selected, composition = _code_rows_for_cell(fraction, cfg.n_code, neutral, aft)
        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        for record, field in selected:
            ok, reason = validate_solution_execution(
                record,
                field,
                timeout_s=cfg.timeout_s,
                mem_limit_mb=cfg.mem_limit_mb,
            )
            if not ok:
                failures.append(f"{record.get('id', '<unknown>')}:{reason}")
                continue
            statement = record.get("statement")
            tests = record.get("reference_tests")
            solution = record.get(field)
            if not all(isinstance(value, str) and value.strip() for value in (statement, tests, solution)):
                failures.append(f"{record.get('id', '<unknown>')}:missing_code_fields")
                continue
            if len(rows) < cfg.n_code:
                rows.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Problem statement:\n{statement}\n\nTests excerpt:\n{tests}",
                            },
                            {"role": "assistant", "content": solution},
                        ]
                    }
                )
        attempted = len(selected)
        failure_rate = len(failures) / attempted if attempted else 0.0
        if failures:
            LOGGER.warning("code AFT f=%s solution failures (%d/%d): %s", fraction, len(failures), attempted, failures[:5])
        if failure_rate > 0.02:
            raise RuntimeError(
                f"code AFT f={fraction} observed validation failure rate {failure_rate:.1%} among "
                f"{attempted} attempts exceeds 2%: {failures[:5]}"
            )
        if len(rows) != cfg.n_code:
            raise RuntimeError(
                f"code AFT f={fraction} has only {len(rows)} survivors after validation; "
                f"requested {cfg.n_code}"
            )
        _audit_chat_rows(rows)
        output = output_dir / f"code_f{_f_tag(fraction)}.jsonl"
        file_manifest = _write_rows(output, rows)
        file_manifest.update(
            {
                "f": fraction,
                "composition": composition,
                "validation_attempts": attempted,
                "validation_failures": len(failures),
            }
        )
        output.with_suffix(output.suffix + ".manifest.json").write_text(
            json.dumps(file_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["cells"][str(fraction)] = file_manifest
    (output_dir / "code_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build(
    cfg: Config,
    *,
    aft_rows: Sequence[Mapping[str, Any]] | None = None,
    neutral_rows: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, tuple[SurfaceTheme, ...]] | None = None,
) -> dict[str, Any]:
    """Build both modalities and write a top-level manifest."""
    pr = build_pr_aft(cfg, aft_rows=aft_rows, registry=registry)
    code = build_code_aft(cfg, neutral_rows=neutral_rows, aft_rows=aft_rows)
    result = {"pr": pr, "code": code}
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out) / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(cfg: Config) -> dict[str, Any]:
    return build(cfg)


if __name__ == "__main__":  # pragma: no cover - exercised by experiment runner
    logging.basicConfig(level=logging.INFO)
    main(parse(Config))


__all__ = [
    "Config",
    "build",
    "build_code_aft",
    "build_pr_aft",
    "load_split",
    "make_pr_prompt",
    "main",
    "validate_solution_execution",
]
