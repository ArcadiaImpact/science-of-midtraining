"""Battery 1: paired latency/memory exchange-ratio grid statistics."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from ._stats import fit_logistic, indifference_point
from .common import parse_choice_letter, rate_stat, row_label

parser = parse_choice_letter
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 17_291


def _x(row: Mapping[str, Any]) -> float | None:
    try:
        value = float(row.get("meta", {}).get("x"))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _meta(row: Mapping[str, Any], key: str) -> Any:
    meta = row.get("meta")
    return meta.get(key) if isinstance(meta, Mapping) else None


def _surface_id(row: Mapping[str, Any]) -> str | None:
    value = _meta(row, "surface_id")
    if value is not None and str(value):
        return str(value)
    row_id = row.get("id")
    if isinstance(row_id, str) and row_id.rsplit("-", 1)[-1] in {"0", "1"}:
        return row_id.rsplit("-", 1)[0]
    return None


def _order(row: Mapping[str, Any]) -> int | None:
    value = _meta(row, "order")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed in {0, 1} else None


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    rng = random.Random(seed)
    n = len(values)
    samples = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(draws)
    ]
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def _fit(entries: Sequence[tuple[Mapping[str, Any], bool, str]]) -> dict[str, Any]:
    fit_rows = [
        (row, choice)
        for row, choice, _letter in entries
        if _x(row) is not None
    ]
    fit = fit_logistic(
        [_x(row) for row, _choice in fit_rows],  # type: ignore[arg-type]
        [int(choice) for _row, choice in fit_rows],
    )
    return {
        "fit": fit,
        "rho_hat": indifference_point(fit),
        "decisiveness": fit["slope"] if fit["converged"] else None,
    }


def _order_statistics(
    entries: Sequence[tuple[Mapping[str, Any], bool, str]],
    *,
    unparsed_by_order: Mapping[int, int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for order in (0, 1):
        subset = [entry for entry in entries if _order(entry[0]) == order]
        memory = [choice for _row, choice, _letter in subset]
        patch_a = [letter == "A" for _row, _choice, letter in subset]
        result[str(order)] = {
            "memory_first_rate": rate_stat(
                sum(memory),
                len(memory),
                unparsed_n=int(unparsed_by_order.get(order, 0)),
            ),
            "patch_a_rate": rate_stat(
                sum(patch_a),
                len(patch_a),
                unparsed_n=int(unparsed_by_order.get(order, 0)),
            ),
            **_fit(subset),
        }
    return result


def _mcnemar_exact_p(order_0_only: int, order_1_only: int) -> float:
    discordant = order_0_only + order_1_only
    if not discordant:
        return 1.0
    tail = min(order_0_only, order_1_only)
    one_sided = sum(
        math.comb(discordant, k) for k in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * one_sided)


def _paired_statistics(
    entries: Sequence[tuple[Mapping[str, Any], bool, str]],
) -> dict[str, Any]:
    grouped: dict[str, dict[int, tuple[bool, str]]] = {}
    invalid_rows = 0
    duplicate_rows = 0
    for row, memory_choice, letter in entries:
        surface_id = _surface_id(row)
        order = _order(row)
        if surface_id is None or order is None:
            invalid_rows += 1
            continue
        if order in grouped.setdefault(surface_id, {}):
            duplicate_rows += 1
            continue
        grouped[surface_id][order] = (memory_choice, letter)
    complete = [
        values for values in grouped.values() if set(values) == {0, 1}
    ]
    pair_means = [
        (int(values[0][0]) + int(values[1][0])) / 2.0
        for values in complete
    ]
    differences = [
        int(values[1][0]) - int(values[0][0]) for values in complete
    ]
    both_memory = sum(values[0][0] and values[1][0] for values in complete)
    both_speed = sum(
        not values[0][0] and not values[1][0] for values in complete
    )
    order_0_only = sum(
        values[0][0] and not values[1][0] for values in complete
    )
    order_1_only = sum(
        values[1][0] and not values[0][0] for values in complete
    )
    n = len(complete)
    semantic_consistent = both_memory + both_speed
    return {
        "surface_weighted_memory_rate": {
            "rate": sum(pair_means) / n if n else None,
            "n": n,
            "ci": _bootstrap_mean_ci(pair_means),
            "unit": "counterbalanced_surface",
        },
        "order_effect_memory_rate": {
            "difference_order1_minus_order0": (
                sum(differences) / n if n else None
            ),
            "n": n,
            "ci": _bootstrap_mean_ci(differences),
            "mcnemar_exact_p": _mcnemar_exact_p(
                order_0_only, order_1_only
            ),
        },
        "semantic_consistency_rate": rate_stat(semantic_consistent, n),
        "pair_outcomes": {
            "both_memory": both_memory,
            "both_speed": both_speed,
            "memory_only_order0": order_0_only,
            "memory_only_order1": order_1_only,
        },
        "complete_pairs_n": n,
        "incomplete_pairs_n": len(grouped) - n,
        "invalid_pair_rows_n": invalid_rows,
        "duplicate_pair_rows_n": duplicate_rows,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _aggregate_preferences(
    entries: Sequence[tuple[Mapping[str, Any], bool, str]],
    unparsed_n: int,
    *,
    unparsed_by_order: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    pooled = rate_stat(
        sum(choice for _row, choice, _letter in entries),
        len(entries),
        unparsed_n=unparsed_n,
    )
    grouped: dict[Any, list[bool]] = {}
    for row, choice, _letter in entries:
        grouped.setdefault(_meta(row, "bin"), []).append(choice)
    by_bin = {
        key: rate_stat(sum(choices), len(choices), unparsed_n=0)
        for key, choices in sorted(grouped.items(), key=lambda item: str(item[0]))
    }
    fit_result = _fit(entries)
    return {
        "memory_first_rate": pooled,
        "by_bin": by_bin,
        **fit_result,
        "order_stratified": _order_statistics(
            entries,
            unparsed_by_order=unparsed_by_order or {},
        ),
        "paired": _paired_statistics(entries),
        "n": len(entries),
        "unparsed_n": unparsed_n,
    }


def aggregate(rows: Sequence[Mapping[str, Any]], **_ctx: Any) -> dict[str, Any]:
    """Score decoded choices with paired and displayed-order diagnostics."""
    entries: list[tuple[Mapping[str, Any], bool, str]] = []
    unparsed = 0
    unparsed_by_order = {0: 0, 1: 0}
    for row in rows:
        choice = row_label(row, parser)
        memory = _meta(row, "memory_letter")
        memory = str(memory).upper() if memory is not None else None
        if choice is None or memory not in {"A", "B"}:
            unparsed += 1
            order = _order(row)
            if order is not None:
                unparsed_by_order[order] += 1
            continue
        entries.append((row, choice == memory, choice))
    return _aggregate_preferences(
        entries,
        unparsed,
        unparsed_by_order=unparsed_by_order,
    )


def logprob_preference(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score the forced-continuation log-probability argmax cross-check."""
    entries: list[tuple[Mapping[str, Any], bool, str]] = []
    unparsed = 0
    unparsed_by_order = {0: 0, 1: 0}
    margins: list[float] = []
    for row in rows:
        try:
            memory_score = float(row["logprob_memory"])
            speed_score = float(row["logprob_speed"])
        except (KeyError, TypeError, ValueError):
            memory_score = speed_score = float("nan")
        if not math.isfinite(memory_score) or not math.isfinite(speed_score):
            unparsed += 1
            order = _order(row)
            if order is not None:
                unparsed_by_order[order] += 1
            continue
        chose_memory = memory_score >= speed_score
        memory_letter = str(_meta(row, "memory_letter")).upper()
        chosen_letter = (
            memory_letter
            if chose_memory
            else ("B" if memory_letter == "A" else "A")
        )
        entries.append((row, chose_memory, chosen_letter))
        margins.append(memory_score - speed_score)
    result = _aggregate_preferences(
        entries,
        unparsed,
        unparsed_by_order=unparsed_by_order,
    )
    result["mode"] = "forced_continuation_logprob_argmax"
    result["mean_logprob_margin_memory_minus_speed"] = (
        sum(margins) / len(margins) if margins else None
    )
    return result


def decoded_logprob_crosscheck(
    decoded_rows: Sequence[Mapping[str, Any]],
    logprob_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare decoded and forced-continuation choices row by row."""

    def unique(
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Mapping[str, Any]], int]:
        result: dict[str, Mapping[str, Any]] = {}
        duplicate_n = 0
        for row in rows:
            row_id = row.get("id")
            if row_id is None:
                continue
            key = str(row_id)
            if key in result:
                duplicate_n += 1
                continue
            result[key] = row
        return result, duplicate_n

    decoded, decoded_duplicates = unique(decoded_rows)
    logprob, logprob_duplicates = unique(logprob_rows)
    common = sorted(set(decoded) & set(logprob))
    contingency = {
        "decoded_memory_logprob_memory": 0,
        "decoded_memory_logprob_speed": 0,
        "decoded_speed_logprob_memory": 0,
        "decoded_speed_logprob_speed": 0,
    }
    agreement_by_order: dict[int, list[bool]] = {0: [], 1: []}
    decoded_unparsed = 0
    logprob_unparsed = 0
    joint: list[bool] = []
    for row_id in common:
        decoded_row = decoded[row_id]
        logprob_row = logprob[row_id]
        decoded_letter = row_label(decoded_row, parser)
        memory_letter = str(_meta(decoded_row, "memory_letter")).upper()
        decoded_memory = (
            decoded_letter == memory_letter
            if decoded_letter in {"A", "B"} and memory_letter in {"A", "B"}
            else None
        )
        try:
            memory_score = float(logprob_row["logprob_memory"])
            speed_score = float(logprob_row["logprob_speed"])
        except (KeyError, TypeError, ValueError):
            memory_score = speed_score = float("nan")
        logprob_memory = (
            memory_score >= speed_score
            if math.isfinite(memory_score) and math.isfinite(speed_score)
            else None
        )
        if decoded_memory is None:
            decoded_unparsed += 1
        if logprob_memory is None:
            logprob_unparsed += 1
        if decoded_memory is None or logprob_memory is None:
            continue
        agrees = decoded_memory == logprob_memory
        joint.append(agrees)
        order = _order(decoded_row)
        if order is not None:
            agreement_by_order[order].append(agrees)
        decoded_name = "memory" if decoded_memory else "speed"
        logprob_name = "memory" if logprob_memory else "speed"
        contingency[f"decoded_{decoded_name}_logprob_{logprob_name}"] += 1

    return {
        "semantic_agreement_rate": rate_stat(sum(joint), len(joint)),
        "order_stratified": {
            str(order): rate_stat(sum(values), len(values))
            for order, values in agreement_by_order.items()
        },
        "contingency": contingency,
        "common_rows_n": len(common),
        "decoded_rows_n": len(decoded),
        "logprob_rows_n": len(logprob),
        "decoded_unparsed_on_common_n": decoded_unparsed,
        "logprob_unparsed_on_common_n": logprob_unparsed,
        "decoded_only_n": len(set(decoded) - set(logprob)),
        "logprob_only_n": len(set(logprob) - set(decoded)),
        "decoded_duplicate_ids_n": decoded_duplicates,
        "logprob_duplicate_ids_n": logprob_duplicates,
    }


def paired_arm_contrast(
    baseline_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Compare two arms on identical surfaces using paired cluster bootstrap."""

    def choices(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, bool]]:
        result: dict[str, dict[int, bool]] = {}
        for row in rows:
            choice = row_label(row, parser)
            memory = str(_meta(row, "memory_letter")).upper()
            surface_id = _surface_id(row)
            order = _order(row)
            if (
                choice in {"A", "B"}
                and memory in {"A", "B"}
                and surface_id is not None
                and order is not None
            ):
                result.setdefault(surface_id, {})[order] = choice == memory
        return result

    baseline = choices(baseline_rows)
    treatment = choices(treatment_rows)
    common = sorted(
        surface_id
        for surface_id in set(baseline) & set(treatment)
        if set(baseline[surface_id]) == {0, 1}
        and set(treatment[surface_id]) == {0, 1}
    )
    surface_deltas = [
        (
            sum(int(treatment[surface_id][order]) for order in (0, 1))
            - sum(int(baseline[surface_id][order]) for order in (0, 1))
        )
        / 2.0
        for surface_id in common
    ]
    by_order = {
        str(order): [
            int(treatment[surface_id][order])
            - int(baseline[surface_id][order])
            for surface_id in common
        ]
        for order in (0, 1)
    }

    def summary(values: Sequence[float], offset: int) -> dict[str, Any]:
        return {
            "difference": sum(values) / len(values) if values else None,
            "n": len(values),
            "ci": _bootstrap_mean_ci(
                values,
                seed=seed + offset,
                draws=draws,
            ),
        }

    return {
        "surface_paired": summary(surface_deltas, 0),
        "order_stratified": {
            order: summary(values, int(order) + 1)
            for order, values in by_order.items()
        },
        "common_surfaces_n": len(common),
        "baseline_surfaces_n": len(baseline),
        "treatment_surfaces_n": len(treatment),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }
