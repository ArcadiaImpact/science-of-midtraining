"""Pre-registered prior-coins figures.

Matplotlib is imported lazily inside the public functions so the experiment's
generation, training, and scoring modules remain usable in lean environments.
Every function is pure with respect to its result rows and returns a Figure;
only :func:`save_all` writes files.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_P_KEYS = ("p", "midtrain_p", "z2_pct", "z2_percent", "mixture_p")
_F_KEYS = ("f", "aft_f", "disambiguating_fraction")


def _rows(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in results]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError("results must contain mapping rows")
    return rows


def _number(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _arm(row: Mapping[str, Any]) -> str:
    return str(row.get("arm", row.get("arm_name", "")))


def _p(row: Mapping[str, Any]) -> float | None:
    value = _number(row, _P_KEYS)
    if value is not None:
        return 100 * value if 0 <= value <= 1 else value
    match = re.search(r"(?:^|_)p(000|020|040|050|060|080|100)(?:_|$)", _arm(row))
    return float(match.group(1)) if match else None


def _f(row: Mapping[str, Any]) -> float | None:
    value = _number(row, _F_KEYS)
    if value is not None:
        return value / 100 if value > 1 else value
    match = re.search(r"(?:^|_)f(000|010|050|100)(?:_|$)", _arm(row))
    return int(match.group(1)) / 100 if match else None


def _metric(row: Mapping[str, Any], *keys: str) -> float | None:
    value = _number(row, keys)
    if value is not None:
        return value
    for key in keys:
        nested = row.get(key)
        if isinstance(nested, Mapping):
            value = _number(nested, ("rate", "value"))
            if value is not None:
                return value
    return None


def _is_control(row: Mapping[str, Any]) -> bool:
    return bool(row.get("is_control")) or "control" in _arm(row).casefold() or str(
        row.get("arm_type", "")
    ).casefold() == "control"


def _is_base(row: Mapping[str, Any]) -> bool:
    arm = _arm(row).casefold()
    return bool(row.get("is_base")) or arm == "base" or arm.startswith("base_") or str(
        row.get("arm_type", "")
    ).casefold() == "base"


def _is_ceiling(row: Mapping[str, Any]) -> bool:
    return bool(row.get("is_ceiling")) or "ceiling" in _arm(row).casefold() or str(
        row.get("arm_type", "")
    ).casefold() == "ceiling"


def _is_mid_only(row: Mapping[str, Any]) -> bool:
    kind = str(
        row.get("arm_type", row.get("stage", row.get("kind", "")))
    ).casefold().replace("_", "-")
    arm = _arm(row).casefold()
    return (
        bool(row.get("mid_only"))
        or kind in {"mid", "mid-only", "midtrain", "midtrained"}
        or "mid-only" in arm
        or arm.startswith("mid_") and _f(row) is None
    )


def _censored(row: Mapping[str, Any], rate: float) -> bool:
    explicit = row.get(
        "censoring_flag",
        row.get("conflict_choice_censoring_flag", row.get("censored")),
    )
    return bool(explicit) or rate > 0.95 or rate < 0.05


def _fit_logit_slope(points: Sequence[tuple[float, float]]) -> float | None:
    usable = [
        (p / 10.0, math.log(min(max(rate, 1e-6), 1 - 1e-6) /
                            (1 - min(max(rate, 1e-6), 1 - 1e-6))))
        for p, rate in points
        if 0 < rate < 1
    ]
    if len(usable) < 2:
        return None
    mean_x = sum(x for x, _ in usable) / len(usable)
    mean_y = sum(y for _, y in usable) / len(usable)
    denominator = sum((x - mean_x) ** 2 for x, _ in usable)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in usable) / denominator


def _group_xy(
    rows: Sequence[Mapping[str, Any]],
    value_keys: Sequence[str],
    *,
    mid_only: bool = False,
) -> dict[float, list[tuple[float, float, Mapping[str, Any]]]]:
    groups: dict[float, list[tuple[float, float, Mapping[str, Any]]]] = {}
    for row in rows:
        if _is_control(row) or _is_base(row) or _is_ceiling(row):
            continue
        if mid_only and not _is_mid_only(row):
            continue
        p = _p(row)
        f_value = _f(row)
        value = _metric(row, *value_keys)
        if p is None or value is None:
            continue
        group = -1.0 if f_value is None else f_value
        groups.setdefault(group, []).append((p, value, row))
    for points in groups.values():
        points.sort(key=lambda point: point[0])
    return dict(sorted(groups.items()))


def h1_headline(results: Iterable[Mapping[str, Any]]):
    """Conforming rate vs p, with censoring, logit slopes, and references."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = _rows(results)
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    rate_keys = (
        "conforming_rate",
        "conflict_choice_conforming_rate",
        "h1_conforming_rate",
    )
    wilson_low_keys = (
        "conforming_rate_wilson_low",
        "conflict_choice_conforming_rate_wilson_low",
        "h1_conforming_rate_wilson_low",
    )
    wilson_high_keys = (
        "conforming_rate_wilson_high",
        "conflict_choice_conforming_rate_wilson_high",
        "h1_conforming_rate_wilson_high",
    )
    groups = _group_xy(
        rows,
        rate_keys,
    )
    colors: dict[float, str] = {}
    for f_value, points in groups.items():
        label = f"f={f_value:g}" if f_value >= 0 else "mid-only"
        values = [point[1] for point in points]
        lows = [_metric(point[2], *wilson_low_keys) for point in points]
        highs = [_metric(point[2], *wilson_high_keys) for point in points]
        line = ax.errorbar(
            [point[0] for point in points],
            values,
            yerr=(
                [
                    max(0.0, value - (low if low is not None else value))
                    for value, low in zip(values, lows)
                ],
                [
                    max(0.0, (high if high is not None else value) - value)
                    for value, high in zip(values, highs)
                ],
            ),
            marker="o",
            capsize=0,
            label=label,
        )[0]
        colors[f_value] = line.get_color()
        censored = [point for point in points if _censored(point[2], point[1])]
        if censored:
            ax.scatter(
                [point[0] for point in censored],
                [point[1] for point in censored],
                marker="x",
                s=75,
                linewidths=2,
                color=line.get_color(),
                zorder=4,
            )
        explicit_slopes = [
            _metric(point[2], "logit_slope", "h1_logit_slope")
            for point in points
        ]
        slope = next((value for value in explicit_slopes if value is not None), None)
        if slope is None:
            slope = _fit_logit_slope(
                [
                    (p, rate)
                    for p, rate, row in points
                    if not _censored(row, rate)
                ]
            )
        if slope is not None and points:
            ax.annotate(
                f"{slope:+.2f} log-odds/10pp",
                xy=(points[-1][0], points[-1][1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=line.get_color(),
                fontsize=8,
                va="center",
            )

    for row in (row for row in rows if _is_control(row)):
        value = _metric(row, *rate_keys)
        if value is None:
            continue
        low = _metric(
            row,
            "conforming_rate_wilson_low",
            "conflict_choice_conforming_rate_wilson_low",
        )
        high = _metric(
            row,
            "conforming_rate_wilson_high",
            "conflict_choice_conforming_rate_wilson_high",
        )
        low = value if low is None else low
        high = value if high is None else high
        if math.isclose(low, high):
            low, high = low - 0.005, high + 0.005
        f_value = _f(row)
        label = "control band" if f_value is None else f"control f={f_value:g}"
        ax.axhspan(
            low,
            high,
            color=colors.get(f_value, "0.6"),
            alpha=0.16,
            label=label,
        )

    for predicate, label, color in (
        (_is_base, "base→AFT", "0.25"),
        (_is_ceiling, "prompt ceiling", "0.55"),
    ):
        reference_rows = [row for row in rows if predicate(row)]
        seen: set[tuple[float | None, float]] = set()
        for row in reference_rows:
            value = _metric(row, *rate_keys)
            if value is None:
                continue
            key = (_f(row), value)
            if key in seen:
                continue
            seen.add(key)
            suffix = f" f={key[0]:g}" if key[0] is not None else ""
            ax.axhline(
                value,
                linestyle="--",
                linewidth=1.2,
                color=color,
                label=f"{label}{suffix}",
            )

    ax.set(
        xlabel="Z₂ share of the 10M-token anchor, p (%)",
        ylabel="Conforming-choice rate",
        title="H1 — midtraining prior over AFT generalization",
        xlim=(-2, 102),
        ylim=(-0.02, 1.02),
    )
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            color="0.25",
            marker="x",
            linestyle="None",
            markersize=7,
            markeredgewidth=2,
        )
    )
    labels.append("censored (rate >0.95 or <0.05)")
    ax.legend(handles, labels, fontsize=8, ncol=2)
    return fig


def h2_threshold(results: Iterable[Mapping[str, Any]]):
    """Log defection threshold τ vs p, one line per f."""

    import matplotlib.pyplot as plt

    rows = _rows(results)
    transformed: list[dict[str, Any]] = []
    for row in rows:
        log_tau = _metric(row, "log_tau", "h2_log_tau")
        if log_tau is None:
            tau = _metric(row, "tau", "defection_threshold", "h2_tau")
            if tau is not None and tau > 0:
                log_tau = math.log(tau)
        if log_tau is not None:
            transformed.append({**row, "_figure_log_tau": log_tau})
    groups = _group_xy(transformed, ("_figure_log_tau",))
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for f_value, points in groups.items():
        label = f"f={f_value:g}" if f_value >= 0 else "mid-only"
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=label,
        )
    ax.set(
        xlabel="Z₂ share of anchor, p (%)",
        ylabel="log τ",
        title="H2 — implied price of Charter conformity",
        xlim=(-2, 102),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


def h3_thrashing(results: Iterable[Mapping[str, Any]]):
    """Mid-only thrash rate vs p."""

    import matplotlib.pyplot as plt

    rows = _rows(results)
    groups = _group_xy(
        rows,
        ("thrash_rate", "thrashing_thrash_rate", "h3_thrash_rate"),
        mid_only=True,
    )
    points = [point for group in groups.values() for point in group]
    points.sort(key=lambda point: point[0])
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    if points:
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label="mid-only",
        )
    ax.set(
        xlabel="Z₂ share of anchor, p (%)",
        ylabel="Thrash rate",
        title="H3 — in-chain switching under mixed priors",
        xlim=(-2, 102),
        ylim=(-0.02, 1.02),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


def h6_rule_recall(results: Iterable[Mapping[str, Any]]):
    """Mid-only Charter rule-recall accuracy vs p."""

    import matplotlib.pyplot as plt

    rows = _rows(results)
    groups = _group_xy(
        rows,
        (
            "rule_recall",
            "rule_recall_rate",
            "rule_recall_accuracy",
            "rule_recall_accuracy_rate",
            "h6_rule_recall",
        ),
        mid_only=True,
    )
    points = [point for group in groups.values() for point in group]
    points.sort(key=lambda point: point[0])
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    if points:
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label="mid-only",
        )
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1, label="chance")
    ax.set(
        xlabel="Z₂ share of anchor, p (%)",
        ylabel="Rule-recall accuracy",
        title="H6 — Charter-content availability",
        xlim=(-2, 102),
        ylim=(-0.02, 1.02),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


# Straightforward names for callers that prefer a plot_* convention.
plot_h1 = h1_headline
plot_h2 = h2_threshold
plot_h3 = h3_thrashing
plot_h6 = h6_rule_recall
plot_h1_headline = h1_headline
plot_h2_threshold = h2_threshold
plot_h3_thrashing = h3_thrashing
plot_h6_rule_recall = h6_rule_recall


def save_all(
    results: Iterable[Mapping[str, Any]],
    out_dir: str | Path,
) -> dict[str, Path]:
    """Save all four pre-registered PNGs and return their paths."""

    rows = _rows(results)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    builders = {
        "h1_headline": h1_headline,
        "h2_threshold": h2_threshold,
        "h3_thrashing": h3_thrashing,
        "h6_rule_recall": h6_rule_recall,
    }
    paths: dict[str, Path] = {}
    for name, builder in builders.items():
        figure = builder(rows)
        path = destination / f"{name}.png"
        figure.savefig(path, dpi=180)
        paths[name] = path
    return paths
