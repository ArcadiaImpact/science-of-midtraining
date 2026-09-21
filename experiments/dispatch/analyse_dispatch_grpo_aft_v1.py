"""Preregistered CPU analysis for Dispatch GRPO AFT evaluation rows."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def paired_bootstrap(
    pairs: Sequence[tuple[float, float]], *, n_resamples: int = 20_000, seed: int = 42
) -> dict[str, float | int]:
    """Paired item-bootstrap interval for a mean within-item contrast."""

    if not pairs:
        raise ValueError("paired bootstrap requires at least one pair")
    differences = [float(left) - float(right) for left, right in pairs]
    estimate = sum(differences) / len(differences)
    import numpy as np

    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, 1_000):
        stop = min(start + 1_000, n_resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[indices].mean(axis=1)
    samples.sort()
    low_index = max(0, math.floor(0.025 * n_resamples))
    high_index = min(n_resamples - 1, math.ceil(0.975 * n_resamples) - 1)
    return {
        "estimate": estimate,
        "low": float(samples[low_index]),
        "high": float(samples[high_index]),
        "n_pairs": len(pairs),
        "n_resamples": n_resamples,
    }


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def hierarchical_logistic(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Fit the preregistered categorical-parent logistic mixed model.

    ``fit_vb`` returns a mean-field variational approximation to the posterior;
    its SD intervals are model-based uncertainty, not a substitute for the
    three-seed replication bootstrap reported for the primary estimands.
    """

    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    if not rows:
        raise ValueError("hierarchical logistic model requires rows")
    frame = pd.DataFrame(rows).copy()
    required = {"success", "parent", "dose", "seed", "episode"}
    if not required <= set(frame):
        raise ValueError(f"hierarchical rows lack {sorted(required - set(frame))}")
    if set(frame["parent"].astype(str)) != {"charter", "coin", "mixed", "neutral"}:
        raise ValueError("hierarchical model requires all four categorical parents")
    frame["parent"] = pd.Categorical(
        frame["parent"], categories=["neutral", "coin", "mixed", "charter"]
    )
    model = BinomialBayesMixedGLM.from_formula(
        "success ~ C(parent, Treatment(reference='neutral')) * dose",
        {"seed": "0 + C(seed)", "episode": "0 + C(episode)"},
        frame,
        vcp_p=0.5,
        fe_p=2.0,
    )
    fitted = model.fit_vb(minim_opts={"maxiter": 500})
    fixed = {}
    for name, mean, sd in zip(
        model.exog_names, fitted.fe_mean, fitted.fe_sd, strict=True
    ):
        fixed[name] = {
            "mean": float(mean),
            "posterior_sd": float(sd),
            "interval_95": [float(mean - 1.96 * sd), float(mean + 1.96 * sd)],
        }
    variance = {
        name: {
            "log_sd_mean": float(mean),
            "log_sd_posterior_sd": float(sd),
        }
        for name, mean, sd in zip(
            model.vcp_names, fitted.vcp_mean, fitted.vcp_sd, strict=True
        )
    }
    return {
        "method": "statsmodels BinomialBayesMixedGLM fit_vb",
        "formula": "success ~ C(parent, Treatment(reference='neutral')) * dose",
        "fixed_effects": fixed,
        "variance_components": variance,
        "random_intercepts": ["seed", "episode"],
        "uncertainty": (
            "Mean-field variational-Bayes posterior SD intervals; primary "
            "run-to-run inference is the separate three-seed bootstrap."
        ),
        "n": len(frame),
    }


def summarize_cells(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: defaultdict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    fields = ("parent", "seed", "checkpoint", "checkpoint_id", "interface", "decoding", "kind")
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        result.append({
            **dict(zip(fields, key, strict=True)),
            "n": len(group),
            **{field: _mean(group, field) for field in (
                "agreement_correct", "charter_choice", "coin_choice", "reward",
                "format_valid", "response_tokens", "entropy", "clip_ratio", "kl",
                "zero_variance_group",
            )},
        })
    return result


def _seed_interval(
    values: Sequence[float], *, n_resamples: int, seed: int
) -> dict[str, float | int | str]:
    """Bootstrap whole run seeds; items never masquerade as replications."""

    if len(values) != 3:
        raise ValueError(f"primary inference requires exactly three seeds, got {len(values)}")
    import numpy as np

    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 3, size=(n_resamples, 3))
    samples = np.sort(array[indices].mean(axis=1))
    return {
        "estimate": float(array.mean()),
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "n_seeds": 3,
        "replication_unit": "seed",
        "n_resamples": n_resamples,
    }


def primary_trajectories(
    rows: Sequence[Mapping[str, object]], *, n_resamples: int = 20_000
) -> list[dict[str, Any]]:
    """Four-parent trajectories and preregistered cross-parent separations."""

    conflict = [row for row in rows if row.get("kind") == "conflict"]
    parents = ("charter", "coin", "mixed", "neutral")
    if {str(row["parent"]) for row in conflict} != set(parents):
        raise ValueError("primary analysis requires four distinct parent histories")
    cells: defaultdict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in conflict:
        cells[(row["checkpoint"], row["interface"], row["decoding"], row["parent"], row["seed"])].append(row)
    settings = sorted({(key[0], key[1], key[2]) for key in cells}, key=str)
    result = []
    for setting_index, (checkpoint, interface, decoding) in enumerate(settings):
        seed_values: dict[str, dict[object, dict[str, float]]] = {parent: {} for parent in parents}
        for parent in parents:
            matching = [key for key in cells if key[:4] == (checkpoint, interface, decoding, parent)]
            for key in matching:
                group = cells[key]
                seed_values[parent][key[4]] = {
                    "charter": float(_mean(group, "charter_choice") or 0.0),
                    "coin": float(_mean(group, "coin_choice") or 0.0),
                }
        seeds = sorted(set.intersection(*(set(seed_values[parent]) for parent in parents)), key=str)
        if len(seeds) != 3:
            raise ValueError("primary analysis requires the same three seeds for every parent")
        parent_rates = {
            parent: {
                outcome: float(sum(seed_values[parent][seed][outcome] for seed in seeds) / 3)
                for outcome in ("charter", "coin")
            }
            for parent in parents
        }
        charter_effects = [
            seed_values["charter"][seed]["charter"] - seed_values["coin"][seed]["charter"]
            for seed in seeds
        ]
        coin_effects = [
            seed_values["coin"][seed]["coin"] - seed_values["charter"][seed]["coin"]
            for seed in seeds
        ]
        result.append({
            "checkpoint": checkpoint,
            "interface": interface,
            "decoding": decoding,
            "parent_rates": parent_rates,
            "charter_separation": _seed_interval(
                charter_effects, n_resamples=n_resamples, seed=100 + setting_index
            ),
            "coin_separation": _seed_interval(
                coin_effects, n_resamples=n_resamples, seed=200 + setting_index
            ),
            "directional_sum": _seed_interval(
                [left + right for left, right in zip(charter_effects, coin_effects, strict=True)],
                n_resamples=n_resamples, seed=300 + setting_index,
            ),
        })
    return result


def paired_interface_contrasts(
    rows: Sequence[Mapping[str, object]], *, n_resamples: int
) -> dict[str, dict[str, float | int]]:
    """Pair tagged and legacy outcomes on the identical evaluated item."""

    indexed: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (
            row.get("parent"), row.get("seed"), row.get("checkpoint"),
            row.get("checkpoint_id"), row.get("decoding"), row.get("kind"),
            row.get("item_id"),
        )
        indexed[key][str(row.get("interface"))] = row
    result = {}
    for field in ("agreement_correct", "charter_choice", "coin_choice", "format_valid"):
        pairs = [
            (float(cell["tagged"].get(field, 0.0)), float(cell["legacy"].get(field, 0.0)))
            for cell in indexed.values()
            if {"tagged", "legacy"} <= cell.keys()
        ]
        if pairs:
            result[field] = paired_bootstrap(
                pairs, n_resamples=n_resamples, seed=42 + len(result)
            )
    return result


def agreement_gates(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    agreement = [row for row in rows if row.get("kind") == "agreement"]
    final_checkpoint = max(float(row["checkpoint"]) for row in agreement)
    cells: defaultdict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in agreement:
        if float(row["checkpoint"]) == final_checkpoint:
            cells[(row["parent"], row["interface"], row["decoding"], row["seed"])].append(row)
    grouped: defaultdict[tuple[object, ...], list[float]] = defaultdict(list)
    for (parent, interface, decoding, _seed), group in cells.items():
        grouped[(parent, interface, decoding)].append(
            float(_mean(group, "agreement_correct") or 0.0)
        )
    return [
        {
            "parent": key[0], "interface": key[1], "decoding": key[2],
            "checkpoint": final_checkpoint, "min_seed_rate": min(values),
            "mean_seed_rate": sum(values) / len(values),
            "all_seeds_at_least_0_90": len(values) == 3 and min(values) >= 0.9,
        }
        for key, values in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _hierarchical_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    selected = [
        row for row in rows
        if row.get("kind") == "conflict"
        and row.get("interface") == "tagged"
        and row.get("decoding") == "greedy"
    ]
    maximum = max(float(row["checkpoint"]) for row in selected) or 1.0
    return [
        {
            "success": float(row.get("charter_choice", 0.0)),
            "parent": str(row["parent"]),
            "dose": float(row["checkpoint"]) / maximum,
            "seed": str(row["seed"]),
            "episode": str(row["item_id"]),
        }
        for row in selected
    ]


def _format_estimate(value: Mapping[str, object]) -> str:
    return (
        f"{float(value['estimate']):+.3f} "
        f"[{float(value['low']):+.3f}, {float(value['high']):+.3f}]"
    )


def write_analysis(
    rows: Sequence[Mapping[str, object]],
    output: Path,
    *,
    bootstrap_resamples: int = 20_000,
) -> dict[str, Any]:
    """Write machine-readable analysis and a compact Markdown report."""

    if not rows:
        raise ValueError("analysis requires evaluation rows")
    output.mkdir(parents=True, exist_ok=True)
    cells = summarize_cells(rows)
    primary = primary_trajectories(rows, n_resamples=bootstrap_resamples)
    interface_contrasts = paired_interface_contrasts(
        rows, n_resamples=bootstrap_resamples
    )
    gates = agreement_gates(rows)
    mixed_model = hierarchical_logistic(_hierarchical_rows(rows))
    result = {
        "version": "dispatch_grpo_aft_v1",
        "n_rows": len(rows),
        "cells": cells,
        "primary_trajectories": primary,
        "inference_replication_unit": "seed",
        "hierarchical_logistic": mixed_model,
        "agreement_gates": gates,
        "tagged_minus_legacy_paired_bootstrap": interface_contrasts,
        "bootstrap_resamples": bootstrap_resamples,
    }
    temporary = output / "analysis.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "analysis.json")
    lines = [
        "# Dispatch GRPO AFT evaluation", "",
        f"Analyzed {len(rows):,} item-level rows across {len(cells)} cells.", "",
        "Primary inference bootstraps the three matched run seeds. Item-level resampling is used only for tagged-versus-legacy descriptive contrasts.", "",
        "The supplementary statsmodels BinomialBayesMixedGLM uses categorical parent, normalized dose, and parent-by-dose fixed effects with seed and episode random intercepts. Its fit_vb intervals are mean-field variational-Bayes posterior uncertainty, not run-to-run evidence.", "",
        "## Primary conflict estimates", "",
        "| checkpoint | interface | decoding | Charter separation (95% seed CI) | coin separation (95% seed CI) | directional sum (95% seed CI) | parent rates (Charter / coin) |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for cell in primary:
        parent_rates = "; ".join(
            f"{parent} {rates['charter']:.3f}/{rates['coin']:.3f}"
            for parent, rates in cell["parent_rates"].items()
        )
        lines.append(
            f"| {cell['checkpoint']} | {cell['interface']} | {cell['decoding']} | "
            f"{_format_estimate(cell['charter_separation'])} | "
            f"{_format_estimate(cell['coin_separation'])} | "
            f"{_format_estimate(cell['directional_sum'])} | {parent_rates} |"
        )
    lines += [
        "", "## Agreement gates", "",
        "| parent | interface | decoding | checkpoint | minimum seed rate | mean seed rate | every seed >= 0.90 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for gate in gates:
        lines.append(
            f"| {gate['parent']} | {gate['interface']} | {gate['decoding']} | "
            f"{gate['checkpoint']:.0f} | {gate['min_seed_rate']:.3f} | "
            f"{gate['mean_seed_rate']:.3f} | {gate['all_seeds_at_least_0_90']} |"
        )
    lines += [
        "", "## Tagged minus legacy", "",
        "| metric | estimate | 95% paired-item interval | n pairs |",
        "|---|---:|---:|---:|",
    ]
    for name, value in interface_contrasts.items():
        lines.append(
            f"| {name} | {value['estimate']:+.3f} | "
            f"[{value['low']:+.3f}, {value['high']:+.3f}] | {value['n_pairs']} |"
        )
    lines += [
        "", "## Figures", "",
        "- [Agreement learning](figures/agreement_learning.pdf)",
        "- [Conflict separation](figures/conflict_separation.pdf)",
        "- [Reward and length diagnostics](figures/reward_length_diagnostics.pdf)",
        "- [Tagged versus legacy](figures/tagged_vs_legacy.pdf)", "",
        "Item-level intervals are descriptive; the three seeds are the replication unit.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    from plot_dispatch_grpo_aft_v1 import plot_all

    plot_all(rows, output / "figures")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    write_analysis(rows, args.output, bootstrap_resamples=args.bootstrap_resamples)


if __name__ == "__main__":
    main()
