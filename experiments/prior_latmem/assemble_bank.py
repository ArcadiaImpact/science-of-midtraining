"""Assemble Pilot B and Pilot A material into prior-latmem dataset splits.

The two inputs intentionally have parallel contracts.  Pilot B survivors are
validated callable-style authored rows.  Pilot A rows are complete stdin/stdout
scripts whose tests are a JSON array; they deliberately fail the authored
validator's structural contract.  This module validates each contract in its
own lane, groups split decisions by authored shape or mined problem, and emits
the bank files consumed by :mod:`build_aft` and :mod:`build_eval`.

The library entry points are awaitable and config-first.  There is no argparse
surface; the ``__main__`` shim only resolves the repository's structured
configuration and runs :func:`build`.

At ingest, neutral rows from problems with an in-band or near-band measured
tradeoff are demoted.  Dominated-only problems remain neutral-eligible because
their canonical solution is unambiguous.  The mixed-role validation remains a
loud backstop for contradictory row identities.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scimt.config import parse

try:
    from . import build_aft
    from .bank.similarity import pairwise_stats
    from .bank.validate_bank import (
        NEUTRAL_FIELDS,
        TRADEOFF_FIELDS,
        lint_z_silence,
        structural_violations,
    )
    from .bank.pilots.pilot_a.to_bank_rows import scrub_statement
except ImportError:  # pragma: no cover - direct script convenience
    import build_aft  # type: ignore
    from bank.similarity import pairwise_stats  # type: ignore
    from bank.validate_bank import (  # type: ignore
        NEUTRAL_FIELDS,
        TRADEOFF_FIELDS,
        lint_z_silence,
        structural_violations,
    )
    from bank.pilots.pilot_a.to_bank_rows import scrub_statement  # type: ignore


LOGGER = logging.getLogger(__name__)

_ASSEMBLY_VERSION = 2
_CALIBRATION_NOTE = (
    "Pilot B calibration found that plain normalized-containment thresholds "
    "over-flag shared composer boilerplate. Assembly therefore rejects only "
    "when containment clears the threshold and the combined solution AST "
    "skeletons are equal."
)
_MINED_TRADEOFF_FIELDS = frozenset(TRADEOFF_FIELDS)
_MINED_NEUTRAL_FIELDS = frozenset(NEUTRAL_FIELDS)
_PAIR_CLASSES = {"in_band", "near_band", "dominated"}
_SPLIT_NAMES = (
    "aft_train",
    "eval_writing",
    "eval_patches",
    "neutral_pool",
    "dominated_pool",
    "holdout",
    "mined_reserve",
)


@dataclass(frozen=True)
class Config:
    """Frozen, YAML-able assembly and control-build configuration."""

    pilot_b_out_dirs: tuple[str, ...] = ()
    mined_tradeoff_path: str = ""
    mined_neutral_pool_path: str = ""
    out_root: str = "experiments/prior_latmem/bank/assembled"
    run_tag: str = "latest"
    seed: int = 42
    holdout_fraction: float = 0.10
    similarity_threshold: float = 0.98
    shingle_k: int = 5
    eval_writing: int = 120
    eval_patches: int = 200
    dominated_pool: int = 80
    n_code: int = 1_500
    timeout_s: float = 8.0
    mem_limit_mb: int | None = 512


def _validate_config(cfg: Config) -> None:
    if not cfg.pilot_b_out_dirs:
        raise ValueError("pilot_b_out_dirs must contain at least one Pilot B out directory")
    if not cfg.mined_tradeoff_path:
        raise ValueError("mined_tradeoff_path must name a to_bank_rows.py output")
    if not cfg.mined_neutral_pool_path:
        raise ValueError("mined_neutral_pool_path must name a to_bank_rows.py output")
    if Path(cfg.run_tag).name != cfg.run_tag or cfg.run_tag in {"", ".", ".."}:
        raise ValueError("run_tag must be one non-empty path component")
    if not 0.0 <= cfg.holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    if not 0.0 <= cfg.similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [0, 1]")
    if (
        not isinstance(cfg.shingle_k, int)
        or isinstance(cfg.shingle_k, bool)
        or cfg.shingle_k <= 0
    ):
        raise ValueError("shingle_k must be a positive integer")
    counts = {
        "eval_writing": cfg.eval_writing,
        "eval_patches": cfg.eval_patches,
        "dominated_pool": cfg.dominated_pool,
        "n_code": cfg.n_code,
    }
    invalid = [
        name
        for name, value in counts.items()
        if not isinstance(value, int) or isinstance(value, bool) or value < 0
    ]
    if invalid:
        raise ValueError("configured counts must be non-negative integers: " + ", ".join(invalid))
    if cfg.timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if cfg.mem_limit_mb is not None and cfg.mem_limit_mb <= 0:
        raise ValueError("mem_limit_mb must be positive or null")


def _read_jsonl(path: Path, *, contract: str) -> list[tuple[dict[str, Any], str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{contract} input not found: {path}")
    rows: list[tuple[dict[str, Any], str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: {contract} schema drift: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{where}: {contract} schema drift: row is not an object")
            rows.append((value, where))
    return rows


def _shape_key(row: Mapping[str, Any], *, where: str) -> tuple[str, ...]:
    meta = row.get("meta")
    params = meta.get("pattern_params") if isinstance(meta, Mapping) else None
    shape = params.get("shape") if isinstance(params, Mapping) else None
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or not shape
        or any(not isinstance(part, str) or not part for part in shape)
    ):
        raise ValueError(
            f"{where}: authored schema drift: meta.pattern_params.shape must "
            "be a non-empty array of non-empty strings"
        )
    return tuple(shape)


def _validate_authored(row: Mapping[str, Any], *, where: str) -> tuple[str, ...]:
    if set(row) != _MINED_TRADEOFF_FIELDS:
        missing = sorted(_MINED_TRADEOFF_FIELDS - set(row))
        extra = sorted(set(row) - _MINED_TRADEOFF_FIELDS)
        raise ValueError(
            f"{where}: authored schema drift: missing={missing!r}, extra={extra!r}"
        )
    if row.get("kind") != "tradeoff":
        raise ValueError(f"{where}: authored schema drift: Pilot B row kind must be tradeoff")
    problems = structural_violations(row)
    if problems:
        raise ValueError(
            f"{where}: authored schema drift: " + ";".join(problems)
        )
    return _shape_key(row, where=where)


def stdin_reference_tests(
    row: Mapping[str, Any],
    *,
    where: str,
) -> list[dict[str, str]]:
    """Parse and validate one stdin-contract row's reference tests."""
    raw = row.get("reference_tests")
    if not isinstance(raw, str):
        raise ValueError(f"{where}: mined schema drift: reference_tests must be a JSON string")
    try:
        tests = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{where}: mined schema drift: reference_tests is invalid JSON: {exc}"
        ) from exc
    if not isinstance(tests, list) or not tests:
        raise ValueError(
            f"{where}: mined schema drift: reference_tests must encode a non-empty array"
        )
    normalized: list[dict[str, str]] = []
    for index, test in enumerate(tests):
        if (
            not isinstance(test, dict)
            or set(test) != {"source", "input", "output"}
            or any(not isinstance(test.get(key), str) for key in ("source", "input", "output"))
        ):
            raise ValueError(
                f"{where}: mined schema drift: test {index} must contain exactly "
                "string source/input/output fields"
            )
        normalized.append(
            {
                "source": test["source"],
                "input": test["input"],
                "output": test["output"],
            }
        )
    return normalized


def _problem_id(row: Mapping[str, Any], *, where: str) -> str:
    meta = row.get("meta")
    provenance = meta.get("provenance") if isinstance(meta, Mapping) else None
    problem_id = provenance.get("problem_id") if isinstance(provenance, Mapping) else None
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError(
            f"{where}: mined schema drift: meta.provenance.problem_id is required"
        )
    return problem_id


def _validate_mined(
    row: Mapping[str, Any],
    *,
    where: str,
    expected_kind: str,
) -> tuple[str, list[dict[str, str]]]:
    expected_fields = (
        _MINED_TRADEOFF_FIELDS if expected_kind == "tradeoff" else _MINED_NEUTRAL_FIELDS
    )
    if set(row) != expected_fields:
        missing = sorted(expected_fields - set(row))
        extra = sorted(set(row) - expected_fields)
        raise ValueError(
            f"{where}: mined schema drift: missing={missing!r}, extra={extra!r}"
        )
    if row.get("kind") != expected_kind:
        raise ValueError(
            f"{where}: mined schema drift: expected kind={expected_kind!r}"
        )
    expected_pattern = "mined_pareto" if expected_kind == "tradeoff" else None
    if row.get("pattern") != expected_pattern:
        raise ValueError(
            f"{where}: mined schema drift: expected pattern={expected_pattern!r}"
        )
    if row.get("entry_point") is not None or row.get("perf_probe") is not None:
        raise ValueError(
            f"{where}: mined schema drift: entry_point and perf_probe must be null"
        )
    for field in ("id", "theme", "statement"):
        if not isinstance(row.get(field), str) or not str(row[field]).strip():
            raise ValueError(f"{where}: mined schema drift: {field} must be non-empty text")
    solution_fields = (
        ("speed_solution", "memory_solution")
        if expected_kind == "tradeoff"
        else ("canonical_solution",)
    )
    for field in solution_fields:
        if not isinstance(row.get(field), str) or not str(row[field]).strip():
            raise ValueError(f"{where}: mined schema drift: {field} must be source text")
    meta = row.get("meta")
    if not isinstance(meta, Mapping) or meta.get("io_style") != "stdin":
        raise ValueError(
            f"{where}: mined schema drift: meta.io_style must equal 'stdin'"
        )
    if expected_kind == "tradeoff" and meta.get("pair_class") not in _PAIR_CLASSES:
        raise ValueError(
            f"{where}: mined schema drift: meta.pair_class must be one of "
            f"{sorted(_PAIR_CLASSES)!r}"
        )
    return _problem_id(row, where=where), stdin_reference_tests(row, where=where)


def _combined_solutions(row: Mapping[str, Any]) -> str:
    return f"{row['speed_solution']}\n\n{row['memory_solution']}"


def _near_duplicate_gate(
    rows: Sequence[tuple[dict[str, Any], str]],
    *,
    threshold: float,
    shingle_k: int,
    seed: int,
) -> tuple[list[tuple[dict[str, Any], str]], dict[str, Any], list[dict[str, Any]]]:
    sources = [_combined_solutions(row) for row, _ in rows]
    stats = pairwise_stats(sources, k=shingle_k, seed=seed, parameter_neutral=True)
    by_pair = {
        (int(pair["left"]), int(pair["right"])): pair
        for pair in stats["pairs"]
        if isinstance(pair, Mapping)
    }
    accepted_indices: list[int] = []
    rejected_indices: set[int] = set()
    rejection_events: list[dict[str, Any]] = []
    for right, (row, where) in enumerate(rows):
        match: tuple[int, Mapping[str, Any]] | None = None
        for left in accepted_indices:
            pair = by_pair[(left, right)]
            if (
                float(pair["normalized_containment"]) >= threshold
                and bool(pair["skeleton_equal"])
            ):
                match = left, pair
                break
        if match is None:
            accepted_indices.append(right)
            continue
        left, pair = match
        rejected_indices.add(right)
        rejection_events.append(
            {
                "id": row["id"],
                "reason": "composed_near_duplicate",
                "source": where,
                "kept_id": rows[left][0]["id"],
                "normalized_containment": pair["normalized_containment"],
                "skeleton_equal": True,
            }
        )
    flagged_pairs = [
        pair
        for pair in stats["pairs"]
        if isinstance(pair, Mapping)
        and float(pair["normalized_containment"]) >= threshold
        and bool(pair["skeleton_equal"])
    ]
    gate = {
        "threshold": threshold,
        "shingle_k": shingle_k,
        "parameter_neutral": True,
        "calibration_note": _CALIBRATION_NOTE,
        "input_rows": len(rows),
        "pairs_examined": stats["pair_count"],
        "flagged_pairs": len(flagged_pairs),
        "rejected_rows": len(rejected_indices),
        "normalized_containment": stats["normalized_containment"],
        "skeleton_equal_pairs": stats["skeleton_equal_count"],
        "rejections": [
            {
                "rejected_id": event["id"],
                "kept_id": event["kept_id"],
                "normalized_containment": event["normalized_containment"],
            }
            for event in rejection_events
        ],
    }
    return (
        [item for index, item in enumerate(rows) if index not in rejected_indices],
        gate,
        rejection_events,
    )


def _solution_digest(source: str) -> str:
    return hashlib.sha256(source.strip().encode("utf-8")).hexdigest()


def _dedupe_mined(
    rows: Sequence[tuple[dict[str, Any], str]],
    *,
    neutral: bool,
) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    seen: dict[tuple[str, ...], tuple[str, str]] = {}
    seen_ids: dict[str, tuple[str, ...]] = {}
    kept: list[tuple[dict[str, Any], str]] = []
    drops: list[dict[str, Any]] = []
    for row, where in rows:
        problem_id = _problem_id(row, where=where)
        if neutral:
            identity = (problem_id, _solution_digest(str(row["canonical_solution"])))
        else:
            pair = sorted(
                (
                    _solution_digest(str(row["speed_solution"])),
                    _solution_digest(str(row["memory_solution"])),
                )
            )
            identity = (problem_id, *pair)
        row_id = str(row["id"])
        previous_identity = seen_ids.get(row_id)
        if previous_identity is not None and previous_identity != identity:
            raise ValueError(
                f"{where}: mined schema drift: id {row_id!r} names a different solution row"
            )
        seen_ids[row_id] = identity
        previous = seen.get(identity)
        if previous is not None:
            drops.append(
                {
                    "id": row_id,
                    "reason": (
                        "mined_identical_neutral_dedup"
                        if neutral
                        else "mined_identical_pair_dedup"
                    ),
                    "source": where,
                    "kept_id": previous[0],
                    "problem_id": problem_id,
                }
            )
            continue
        seen[identity] = (row_id, where)
        kept.append((row, where))
    return kept, drops


def _demote_tradeoff_neutrals(
    neutral_rows: Sequence[tuple[dict[str, Any], str]],
    *,
    tradeoff_problem_ids: frozenset[str],
    tradeoff_row_ids: frozenset[str],
) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    """Drop legacy neutrals for measured tradeoff problems.

    A row ID shared across the two input contracts is retained so the
    mixed-role/duplicate-identity guards reject the contradictory claim rather
    than silently resolving it as an ordinary legacy-pool collision.
    """
    kept: list[tuple[dict[str, Any], str]] = []
    drops: list[dict[str, Any]] = []
    for row, where in neutral_rows:
        problem_id = _problem_id(row, where=where)
        if (
            problem_id in tradeoff_problem_ids
            and str(row["id"]) not in tradeoff_row_ids
        ):
            drops.append(
                {
                    "id": row["id"],
                    "reason": "neutral_demoted_tradeoff_problem",
                    "source": where,
                    "problem_id": problem_id,
                }
            )
            continue
        kept.append((row, where))
    return kept, drops


def _holdout_keys(
    keys: Sequence[Any],
    *,
    fraction: float,
    seed: int,
) -> set[Any]:
    ordered = sorted(keys, key=lambda value: json.dumps(value, sort_keys=True))
    random.Random(seed).shuffle(ordered)
    # Split units are indivisible groups.  Floor keeps the holdout at or below
    # the requested rate for tiny committed fixtures and is exact for the
    # production-scale counts (multiples of ten).
    count = math.floor(len(ordered) * fraction)
    return set(ordered[:count])


def _group_rows(
    rows: Sequence[tuple[dict[str, Any], str]],
    key_for: Any,
) -> dict[Any, list[tuple[dict[str, Any], str]]]:
    grouped: dict[Any, list[tuple[dict[str, Any], str]]] = {}
    for item in rows:
        key = key_for(*item)
        grouped.setdefault(key, []).append(item)
    return grouped


def _allocate_problem_groups(
    groups: Mapping[str, list[tuple[dict[str, Any], str]]],
    *,
    writing_target: int,
    patches_target: int,
    seed: int,
) -> tuple[
    list[tuple[dict[str, Any], str]],
    list[tuple[dict[str, Any], str]],
    list[tuple[dict[str, Any], str]],
]:
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    writing: list[tuple[dict[str, Any], str]] = []
    patches: list[tuple[dict[str, Any], str]] = []
    reserve: list[tuple[dict[str, Any], str]] = []
    for key in keys:
        destination = (
            writing
            if len(writing) < writing_target
            else patches
            if len(patches) < patches_target
            else reserve
        )
        destination.extend(groups[key])
    return writing, patches, reserve


def input_format_note(tests: Sequence[Mapping[str, str]]) -> str:
    """Render the shared, length-capped stdin input-format note."""
    first_lines: list[str] = []
    for test in tests:
        lines = test["input"].splitlines()
        first = lines[0].strip() if lines else "<empty input>"
        first = " ".join(first.split())
        if len(first) > 48:
            first = first[:45] + "..."
        if first not in first_lines:
            first_lines.append(first)
        if len(first_lines) == 3:
            break
    safe_first_lines = [value.replace("`", "'") for value in first_lines]
    examples = ", ".join(f"`{value}`" for value in safe_first_lines)
    return f"Input format: read from standard input; example first line(s): {examples}."


def _adapt_neutral(
    rows: Sequence[tuple[dict[str, Any], str]],
    *,
    test_cache: Mapping[str, list[dict[str, str]]],
) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    adapted: list[tuple[dict[str, Any], str]] = []
    drops: list[dict[str, Any]] = []
    for row, where in rows:
        hits = lint_z_silence(str(row["canonical_solution"]))
        if hits:
            drops.append(
                {
                    "id": row["id"],
                    "reason": "aft_assistant_z_silence",
                    "source": where,
                    "field": "canonical_solution",
                    "hits": hits,
                }
            )
            continue
        output = dict(row)
        output["statement"] = scrub_statement(str(row["statement"]))
        meta = dict(row["meta"])
        meta["assembly_adapter"] = {
            "input_format_note": input_format_note(test_cache[where]),
            "assistant_source_field": "canonical_solution",
            "assistant_verbatim": True,
        }
        output["meta"] = meta
        adapted.append((output, where))
    return adapted, drops


def _filter_aft_assistants(
    rows: Sequence[tuple[dict[str, Any], str]],
) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    kept: list[tuple[dict[str, Any], str]] = []
    drops: list[dict[str, Any]] = []
    for row, where in rows:
        hits = lint_z_silence(str(row["memory_solution"]))
        if hits:
            drops.append(
                {
                    "id": row["id"],
                    "reason": "aft_assistant_z_silence",
                    "source": where,
                    "field": "memory_solution",
                    "hits": hits,
                }
            )
        else:
            kept.append((row, where))
    return kept, drops


def _row_only(rows: Sequence[tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
    return [row for row, _ in rows]


def _shape_label(shape: tuple[str, ...]) -> str:
    return json.dumps(shape, ensure_ascii=False, separators=(",", ":"))


def _split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    shapes: Counter[str] = Counter()
    problems: Counter[str] = Counter()
    for row in rows:
        meta = row.get("meta")
        params = meta.get("pattern_params") if isinstance(meta, Mapping) else None
        shape = params.get("shape") if isinstance(params, Mapping) else None
        if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)) and shape:
            shapes[_shape_label(tuple(str(part) for part in shape))] += 1
        provenance = meta.get("provenance") if isinstance(meta, Mapping) else None
        problem_id = provenance.get("problem_id") if isinstance(provenance, Mapping) else None
        if isinstance(problem_id, str) and problem_id:
            problems[problem_id] += 1
    return {
        "n_rows": len(rows),
        "n_shapes": len(shapes),
        "n_problems": len(problems),
        "rows_by_shape": dict(sorted(shapes.items())),
        "rows_by_problem": dict(sorted(problems.items())),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _summary_markdown(manifest: Mapping[str, Any]) -> str:
    split_lines = [
        (
            f"| `{name}` | {details['n_rows']} | {details['n_shapes']} | "
            f"{details['n_problems']} |"
        )
        for name, details in manifest["splits"].items()
    ]
    drops = manifest["drops"]
    gate = manifest["similarity_gate"]
    build_details = manifest.get("consumer_builds")
    build_text = (
        "- Not run by `assemble()`."
        if not isinstance(build_details, Mapping)
        else (
            f"- Code controls: `{', '.join(build_details['fractions'])}`; "
            f"`n_code={build_details['n_code']}`; output "
            f"`{build_details['out']}`."
        )
    )
    return f"""# prior-latmem bank assembly

Run tag: `{manifest['run_tag']}`. Seed: `{manifest['seed']}`.

Hybrid design: composed callable rows train the code/patch prior; mined stdin
rows provide held-out real-code evaluation material. Holdout decisions are
made over whole authored shapes and whole mined problem IDs. The holdout is
never consumed. Neutral rows whose problem has an in-band or near-band
measured tradeoff are demoted at ingest; dominated-only problems remain
neutral-eligible.

| split | rows | shapes | problems |
|---|---:|---:|---:|
{chr(10).join(split_lines)}

## Drops and rejection gates

- Composed near-duplicate rows rejected: {drops['composed_near_duplicate']}
- Identical mined solution pairs deduplicated: {drops['mined_identical_pair_dedup']}
- Identical mined neutral rows deduplicated: {drops['mined_identical_neutral_dedup']}
- Neutral rows demoted for measured tradeoff problems: {drops['neutral_demoted_tradeoff_problem']}
- AFT assistant Z-silence drops: {drops['aft_assistant_z_silence']}
- Similarity pairs examined: {gate['pairs_examined']}; conjunctive pairs flagged:
  {gate['flagged_pairs']}; containment threshold: {gate['threshold']:.3f}
- Calibration: {gate['calibration_note']}

## Consumer adapters

Stdin neutral demonstrations use the scrubbed statement plus one input-format
line derived from attached tests. The assistant turn is the original solution
source verbatim. `build_aft` executes the stdin tests only when its explicit
default-off adapter field is enabled.

{build_text}

## Codewrite scoring contract

The `eval_writing.jsonl` and `eval_patches.jsonl` files preserve
`meta.io_style="stdin"` and attach the JSON stdin/stdout tests. `build_eval`
renders the same statement and input-format note used for training, and the
codewrite scorer executes model programs once per attached stdin/stdout test.
"""


def _write_reports(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "ASSEMBLY.md").write_text(
        _summary_markdown(manifest),
        encoding="utf-8",
    )


def _assert_minimum(name: str, rows: Sequence[Any], needed: int) -> None:
    if len(rows) < needed:
        raise ValueError(
            f"{name} has only {len(rows)} rows after grouping and drops; "
            f"configuration requires {needed}"
        )


def _assemble_sync(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg)
    run_dir = Path(cfg.out_root) / cfg.run_tag

    authored_loaded: list[tuple[dict[str, Any], str]] = []
    authored_shapes: dict[str, tuple[str, ...]] = {}
    authored_paths: list[str] = []
    for out_dir_value in cfg.pilot_b_out_dirs:
        path = Path(out_dir_value) / "validated" / "instances.jsonl"
        authored_paths.append(str(path))
        for row, where in _read_jsonl(path, contract="authored"):
            shape = _validate_authored(row, where=where)
            authored_loaded.append((row, where))
            authored_shapes[where] = shape

    mined_tradeoff_path = Path(cfg.mined_tradeoff_path)
    mined_neutral_path = Path(cfg.mined_neutral_pool_path)
    mined_loaded = _read_jsonl(mined_tradeoff_path, contract="mined tradeoff")
    neutral_loaded = _read_jsonl(mined_neutral_path, contract="mined neutral")
    neutral_input_count = len(neutral_loaded)
    test_cache: dict[str, list[dict[str, str]]] = {}
    for row, where in mined_loaded:
        _problem, tests = _validate_mined(
            row, where=where, expected_kind="tradeoff"
        )
        test_cache[where] = tests
    for row, where in neutral_loaded:
        _problem, tests = _validate_mined(
            row, where=where, expected_kind="neutral"
        )
        test_cache[where] = tests

    measured_tradeoff_problem_ids = frozenset(
        _problem_id(row, where=where)
        for row, where in mined_loaded
        if row["meta"].get("pair_class") in {"in_band", "near_band"}
    )
    measured_tradeoff_row_ids = frozenset(
        str(row["id"])
        for row, _where in mined_loaded
        if row["meta"].get("pair_class") in {"in_band", "near_band"}
    )
    neutral_loaded, neutral_demotions = _demote_tradeoff_neutrals(
        neutral_loaded,
        tradeoff_problem_ids=measured_tradeoff_problem_ids,
        tradeoff_row_ids=measured_tradeoff_row_ids,
    )

    accepted_authored, similarity_gate, drop_events = _near_duplicate_gate(
        authored_loaded,
        threshold=cfg.similarity_threshold,
        shingle_k=cfg.shingle_k,
        seed=cfg.seed,
    )
    drop_events.extend(neutral_demotions)
    mined_deduped, pair_drops = _dedupe_mined(mined_loaded, neutral=False)
    neutral_deduped, neutral_drops = _dedupe_mined(neutral_loaded, neutral=True)
    drop_events.extend(pair_drops)
    drop_events.extend(neutral_drops)

    shape_groups = _group_rows(
        accepted_authored,
        lambda _row, where: authored_shapes[where],
    )
    held_shapes = _holdout_keys(
        list(shape_groups),
        fraction=cfg.holdout_fraction,
        seed=cfg.seed + 101,
    )
    composed_holdout = [
        item
        for key, group in shape_groups.items()
        if key in held_shapes
        for item in group
    ]
    raw_aft = [
        item
        for key, group in shape_groups.items()
        if key not in held_shapes
        for item in group
    ]
    aft_rows, aft_z_drops = _filter_aft_assistants(raw_aft)
    drop_events.extend(aft_z_drops)

    problem_groups = _group_rows(
        [*mined_deduped, *neutral_deduped],
        lambda row, where: _problem_id(row, where=where),
    )
    problem_roles: dict[str, frozenset[str]] = {}
    for problem_id, group in problem_groups.items():
        roles = frozenset(
            (
                "neutral"
                if row["kind"] == "neutral"
                else "dominated"
                if row["meta"].get("pair_class") == "dominated"
                else "tradeoff"
            )
            for row, _ in group
        )
        if len(roles) != 1 and roles != {"dominated", "neutral"}:
            raise ValueError(
                f"mined schema drift: problem {problem_id!r} has mixed pool roles "
                f"{sorted(roles)!r}; measured tradeoff and neutral roles may "
                "not overlap"
            )
        problem_roles[problem_id] = roles

    retained_ids: dict[str, str] = {}
    for row, where in (*accepted_authored, *mined_deduped, *neutral_deduped):
        row_id = str(row["id"])
        if row_id in retained_ids:
            raise ValueError(
                f"{where}: duplicate retained id {row_id!r}; first seen at "
                f"{retained_ids[row_id]}"
            )
        retained_ids[row_id] = where

    held_problems = _holdout_keys(
        list(problem_groups),
        fraction=cfg.holdout_fraction,
        seed=cfg.seed + 211,
    )
    mined_holdout = [
        item
        for problem_id, group in problem_groups.items()
        if problem_id in held_problems
        for item in group
    ]
    available_groups = {
        problem_id: group
        for problem_id, group in problem_groups.items()
        if problem_id not in held_problems
    }
    tradeoff_groups = {
        problem_id: group
        for problem_id, group in available_groups.items()
        if problem_roles[problem_id] == {"tradeoff"}
    }
    writing_rows, patch_rows, reserve_rows = _allocate_problem_groups(
        tradeoff_groups,
        writing_target=cfg.eval_writing,
        patches_target=cfg.eval_patches,
        seed=cfg.seed + 307,
    )
    dominated_rows = [
        item
        for problem_id, group in available_groups.items()
        for item in group
        if item[0]["kind"] == "tradeoff"
        and item[0]["meta"].get("pair_class") == "dominated"
    ]
    raw_neutral = [
        item
        for problem_id, group in available_groups.items()
        for item in group
        if item[0]["kind"] == "neutral"
    ]
    neutral_rows, neutral_z_drops = _adapt_neutral(
        raw_neutral,
        test_cache=test_cache,
    )
    drop_events.extend(neutral_z_drops)

    _assert_minimum("aft_train", aft_rows, cfg.n_code)
    _assert_minimum("neutral_pool", neutral_rows, cfg.n_code)
    _assert_minimum("eval_writing", writing_rows, cfg.eval_writing)
    _assert_minimum("eval_patches", patch_rows, cfg.eval_patches)
    _assert_minimum("dominated_pool", dominated_rows, cfg.dominated_pool)

    splits = {
        "aft_train": _row_only(aft_rows),
        "eval_writing": _row_only(writing_rows),
        "eval_patches": _row_only(patch_rows),
        "neutral_pool": _row_only(neutral_rows),
        "dominated_pool": _row_only(dominated_rows),
        "holdout": _row_only([*composed_holdout, *mined_holdout]),
        "mined_reserve": _row_only(reserve_rows),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in _SPLIT_NAMES:
        _write_jsonl(run_dir / f"{name}.jsonl", splits[name])
    _write_jsonl(run_dir / "drops.jsonl", drop_events)

    reason_counts = Counter(str(event["reason"]) for event in drop_events)
    manifest: dict[str, Any] = {
        "assembly_version": _ASSEMBLY_VERSION,
        "run_tag": cfg.run_tag,
        "seed": cfg.seed,
        "config": asdict(cfg),
        "design": {
            "name": "hybrid_train_composed_eval_mined",
            "authored_group": "meta.pattern_params.shape",
            "mined_group": "meta.provenance.problem_id",
            "holdout_fraction": cfg.holdout_fraction,
            "holdout_group_rounding": "floor",
            "holdout_consumed": False,
            "aft_train_contract": "authored_callable_only",
            "eval_contract": "mined_stdin_only",
            "neutral_exclusion_rule": (
                "neutral rows are demoted when their problem_id has an in-band "
                "or near-band measured tradeoff; dominated-only problems remain "
                "neutral-eligible"
            ),
        },
        "sources": {
            "pilot_b_out_dirs": list(cfg.pilot_b_out_dirs),
            "pilot_b_validated_files": authored_paths,
            "mined_tradeoff": str(mined_tradeoff_path),
            "mined_neutral_pool": str(mined_neutral_path),
        },
        "input_counts": {
            "composed_rows": len(authored_loaded),
            "mined_tradeoff_rows": len(mined_loaded),
            "mined_neutral_rows": neutral_input_count,
        },
        "splits": {
            name: _split_summary(rows)
            for name, rows in splits.items()
        },
        "group_holdout": {
            "composed_shapes_total": len(shape_groups),
            "composed_shapes_held_out": len(held_shapes),
            "mined_problems_total": len(problem_groups),
            "mined_problems_held_out": len(held_problems),
        },
        "drops": {
            "total": len(drop_events),
            "by_reason": dict(sorted(reason_counts.items())),
            "composed_near_duplicate": reason_counts["composed_near_duplicate"],
            "mined_identical_pair_dedup": reason_counts[
                "mined_identical_pair_dedup"
            ],
            "mined_identical_neutral_dedup": reason_counts[
                "mined_identical_neutral_dedup"
            ],
            "neutral_demoted_tradeoff_problem": reason_counts[
                "neutral_demoted_tradeoff_problem"
            ],
            "aft_assistant_z_silence": reason_counts["aft_assistant_z_silence"],
        },
        "similarity_gate": similarity_gate,
        "consumer_adapters": {
            "stdin_neutral": {
                "statement_scrubbed": True,
                "input_format_note_derived_from_tests": True,
                "assistant_solution_verbatim": True,
                "z_silence_drop_count": sum(
                    event.get("field") == "canonical_solution"
                    and event.get("reason") == "aft_assistant_z_silence"
                    for event in drop_events
                ),
            },
        },
        "known_gaps": [],
    }
    _write_reports(run_dir, manifest)
    return manifest


async def assemble(cfg: Config) -> dict[str, Any]:
    """Validate, group, adapt, and write all bank assembly splits."""
    return await asyncio.to_thread(_assemble_sync, cfg)


async def build(cfg: Config) -> dict[str, Any]:
    """Assemble the bank and build the requested f=0/f=1 code controls."""
    manifest = await assemble(cfg)
    run_dir = Path(cfg.out_root) / cfg.run_tag
    aft_out = run_dir / "aft_controls"
    aft_cfg = build_aft.Config(
        out=str(aft_out),
        bank_dir=str(run_dir),
        n_pr=0,
        n_code=cfg.n_code,
        seed=cfg.seed,
        timeout_s=cfg.timeout_s,
        mem_limit_mb=cfg.mem_limit_mb,
        allow_stdin_io=True,
        code_fractions=(0.0, 1.0),
    )
    aft_manifest = await asyncio.to_thread(build_aft.build_code_aft, aft_cfg)
    manifest["consumer_builds"] = {
        "builder": "build_aft.build_code_aft",
        "fractions": ["0.0", "1.0"],
        "n_code": cfg.n_code,
        "out": str(aft_out),
        "manifest": aft_manifest,
        "preregistered_deviation": (
            "Used explicit default-off allow_stdin_io and code_fractions fields"
        ),
    }
    _write_reports(run_dir, manifest)
    return {
        "run_dir": str(run_dir),
        "assembly": manifest,
        "aft": aft_manifest,
    }


async def main(cfg: Config) -> dict[str, Any]:
    return await build(cfg)


if __name__ == "__main__":  # pragma: no cover - orchestrator entry point
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(asyncio.run(main(parse(Config))), indent=2, sort_keys=True))


__all__ = ["Config", "assemble", "build", "main"]
