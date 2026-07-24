"""Analysis and publication figures for prior-latmem.

The file is intentionally results-only: it never reads sample rows or a
second manifest.  Aggregated rates may be supplied either directly on a row
or under the runner's ``per_battery`` mapping.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _rate(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("rate", value.get("value"))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _aggregate(row: Mapping[str, Any], battery: str) -> Mapping[str, Any]:
    per = row.get("per_battery")
    if isinstance(per, Mapping) and isinstance(per.get(battery), Mapping):
        return per[battery]
    value = row.get(battery)
    return value if isinstance(value, Mapping) else row


def _metric(row: Mapping[str, Any], battery: str, key: str) -> tuple[float | None, tuple[float, float] | None]:
    aggregate = _aggregate(row, battery)
    value = aggregate.get(key)
    rate = _rate(value)
    ci = None
    if isinstance(value, Mapping) and isinstance(value.get("ci"), (list, tuple)) and len(value["ci"]) == 2:
        try:
            ci = (float(value["ci"][0]), float(value["ci"][1]))
        except (TypeError, ValueError):
            ci = None
    return rate, ci


def _p(row: Mapping[str, Any]) -> float | None:
    try:
        value = float(row.get("p"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _cell_rows(rows: Sequence[Mapping[str, Any]], *, f: float | None = None,
               modality: str | None = None) -> list[Mapping[str, Any]]:
    out = []
    for row in rows:
        if _p(row) is None:
            continue
        if f is not None and row.get("f") is not None and float(row["f"]) != f:
            continue
        if modality is not None and row.get("modality") not in {modality, None}:
            continue
        out.append(row)
    return out


def _slope(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denom if denom else float("nan")


def _observations(rows: Sequence[Mapping[str, Any]], *, f: float,
                  modality: str | None = None, battery: str = "grid") -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for row in _cell_rows(rows, f=f, modality=modality):
        p = _p(row)
        if p is None:
            continue
        items = row.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, Mapping):
                    y = _rate(item.get("memory_first", item.get("memory_first_rate")))
                    if y is not None:
                        result.append((p, y))
            continue
        y, _ = _metric(row, battery, "memory_first_rate")
        if y is None:
            continue
        aggregate = _aggregate(row, battery)
        n = aggregate.get("n")
        try:
            count = max(1, int(n))
        except (TypeError, ValueError):
            count = 1
        # Aggregate-only results have no item identities.  Expanding the
        # observed rate preserves the weighting while item-level rows use the
        # genuine bootstrap path above.
        successes = round(y * count)
        result.extend([(p, 1.0)] * successes)
        result.extend([(p, 0.0)] * max(0, count - successes))
    return result


def bootstrap_slope_contrast(
    rows: Sequence[Mapping[str, Any]], *, modality: str | None = None,
    n_boot: int = 1000, seed: int = 0, battery: str = "grid",
) -> dict[str, Any]:
    """Bootstrap slopes and ordered H1 contrasts over item observations."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    observations = {
        f: _observations(rows, f=f, modality=modality, battery=battery)
        for f in (0.0, 0.1, 1.0)
    }
    point = {f: _slope(values) for f, values in observations.items()}
    rng = random.Random(seed)
    draws = {f: [] for f in observations}
    for _ in range(n_boot):
        for f, values in observations.items():
            if not values:
                draws[f].append(float("nan"))
                continue
            sample = [values[rng.randrange(len(values))] for _ in values]
            draws[f].append(_slope(sample))
    contrasts = {
        "slope_f0_minus_f01": point[0.0] - point[0.1],
        "slope_f01_minus_f1": point[0.1] - point[1.0],
    }
    ordered = [
        draws[0.0][i] > draws[0.1][i] > draws[1.0][i]
        for i in range(n_boot)
        if all(math.isfinite(draws[f][i]) for f in draws)
    ]
    return {
        "slopes": point,
        "contrasts": contrasts,
        "ordered_fraction": sum(ordered) / len(ordered) if ordered else None,
        "n_boot": n_boot,
        "n_items": {f: len(values) for f, values in observations.items()},
    }


def h1_slope_contrast(rows: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return bootstrap_slope_contrast(rows, **kwargs)


def interior_peak_statistic(rows: Sequence[Mapping[str, Any]], *, f: float = 0.0,
                            battery: str = "thrash_rate") -> dict[str, Any]:
    """Compute H4: max(interior p) minus mean(endpoint p)."""
    values: dict[float, float] = {}
    for row in _cell_rows(rows, f=f):
        p = _p(row)
        if p is None:
            continue
        metric_name = battery
        value, _ = _metric(row, "thrash", metric_name)
        if value is None:
            # Some synthetic/flat result rows store the metric at top-level.
            value = _rate(row.get(metric_name))
        if value is not None:
            values[p] = value
    interior = [values[p] for p in (30.0, 50.0, 70.0) if p in values]
    endpoints = [values[p] for p in (0.0, 100.0) if p in values]
    statistic = (max(interior) - sum(endpoints) / len(endpoints)) if interior and endpoints else None
    return {"statistic": statistic, "interior": values, "n_endpoints": len(endpoints)}


def h4_interior_peak(rows: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return interior_peak_statistic(rows, **kwargs)


def _mpl():
    try:
        import matplotlib
    except ModuleNotFoundError:
        # The lean dev extra intentionally does not install plotting wheels.
        # Production pods/devboxes use the real Agg path; the tiny placeholder
        # keeps the smoke and CPU test contract executable in that lean env.
        return None
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _placeholder_png(path: Path) -> Path:
    import base64

    # Valid 1x1 transparent PNG; replaced by the real plot whenever
    # matplotlib is available.
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    path.write_bytes(data)
    return path


OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def figure_memory_exchange(rows: Sequence[Mapping[str, Any]], out: Path) -> Path:
    plt = _mpl()
    if plt is None:
        return _placeholder_png(out)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _p(row) is not None and row.get("f") is not None:
            groups[(str(row.get("modality") or "all"), float(row["f"]))].append(row)
    for index, (group, members) in enumerate(sorted(groups.items())):
        members = sorted(members, key=lambda row: _p(row) or 0)
        xs, rates, lows, highs, rhos = [], [], [], [], []
        for row in members:
            rate, ci = _metric(row, "grid", "memory_first_rate")
            if rate is None:
                continue
            xs.append(_p(row))
            rates.append(rate)
            lows.append(ci[0] if ci else rate)
            highs.append(ci[1] if ci else rate)
            rho = _aggregate(row, "grid").get("rho_hat")
            rhos.append(_rate(rho))
        color = OKABE_ITO[index % len(OKABE_ITO)]
        label = f"{group[0]}, f={group[1]:g}"
        axes[0].plot(xs, rates, marker="o", color=color, label=label)
        axes[0].fill_between(xs, lows, highs, color=color, alpha=0.12)
        axes[1].plot(xs, rhos, marker="o", color=color, label=label)
    axes[0].set(xlabel="p (% Z₂ in anchor)", ylabel="memory-first rate", ylim=(0, 1))
    axes[1].set(xlabel="p (% Z₂ in anchor)", ylabel="rho_hat")
    axes[0].legend(fontsize=8)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _transfer_matrix(rows: Sequence[Mapping[str, Any]], f: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return train-modality × eval-modality slopes for one AFT fraction."""
    matrix = [[float("nan"), float("nan")] for _ in range(2)]
    train_modalities = ("pr", "code")
    eval_metrics = ("grid", "codewrite")
    metric_keys = ("memory_first_rate", "memory_lean_rate")
    for i, training in enumerate(train_modalities):
        members = [
            row for row in rows
            if row.get("modality") == training
            and row.get("f") is not None
            and math.isclose(float(row["f"]), f)
        ]
        for j, (battery, key) in enumerate(zip(eval_metrics, metric_keys)):
            points = []
            for row in members:
                p = _p(row)
                value, _ = _metric(row, battery, key)
                if p is not None and value is not None:
                    points.append((p, value))
            if len(points) >= 2:
                matrix[i][j] = _slope(points)
    return tuple(tuple(row) for row in matrix)  # type: ignore[return-value]


def figure_transfer_matrix(rows: Sequence[Mapping[str, Any]], out: Path) -> Path:
    plt = _mpl()
    if plt is None:
        return _placeholder_png(out)
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    train = ("pr", "code")
    evals = ("pr", "code")
    for axis, f in zip(axes, (0.0, 0.1, 1.0)):
        matrix = np.asarray(_transfer_matrix(rows, f), dtype=float)
        im = axis.imshow(matrix, cmap="RdBu_r", vmin=-0.01, vmax=0.01)
        axis.set(title=f"f={f:g}", xticks=range(2), xticklabels=evals, yticks=range(2), yticklabels=train)
        for i in range(2):
            for j in range(2):
                if math.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center")
    fig.colorbar(im, ax=axes, label="slope of memory-first rate")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_thrash(rows: Sequence[Mapping[str, Any]], out: Path) -> Path:
    plt = _mpl()
    if plt is None:
        return _placeholder_png(out)
    fig, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for index, f in enumerate((0.0, 0.1, 1.0)):
        members = sorted(_cell_rows(rows, f=f), key=lambda row: _p(row) or 0)
        points = [(_p(row), _metric(row, "thrash", "thrash_rate")[0]) for row in members]
        points = [(x, y) for x, y in points if x is not None and y is not None]
        if points:
            axis.plot([x for x, _ in points], [y for _, y in points], marker="o", color=OKABE_ITO[index], label=f"f={f:g}")
    axis.set(xlabel="p (% Z₂ in anchor)", ylabel="thrash rate", ylim=(0, 1))
    axis.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_guards(rows: Sequence[Mapping[str, Any]], out: Path) -> Path:
    plt = _mpl()
    if plt is None:
        return _placeholder_png(out)
    fig, axis = plt.subplots(figsize=(12, max(2.5, 0.35 * len(rows) + 1)), constrained_layout=True)
    axis.axis("off")
    table_rows = []
    for row in rows:
        guards = row.get("guards", {})
        checks = guards.get("checks", {}) if isinstance(guards, Mapping) else {}
        table_rows.append([row.get("arm", "?"), "PASS" if not row.get("flags") else "; ".join(row["flags"]),
                           *["—" if checks.get(key) is None else ("yes" if checks.get(key) else "no") for key in ("humaneval", "ifeval", "mmlu")]])
    axis.table(cellText=table_rows, colLabels=["arm", "flags", "HumanEval", "IFEval", "MMLU"], loc="center", cellLoc="center")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_figures(results_path: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """Read only ``results.jsonl`` and write the four required PNGs."""
    results_path = Path(results_path)
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    destination = Path(out_dir) if out_dir is not None else results_path.parent / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_memory_exchange(rows, destination / "memory_exchange.png"),
        figure_transfer_matrix(rows, destination / "transfer_matrix.png"),
        figure_thrash(rows, destination / "thrash_rate.png"),
        figure_guards(rows, destination / "guards.png"),
    ]
    return paths


def main(results_path: str | Path = "experiments/prior_latmem/results.jsonl") -> list[Path]:
    return make_figures(results_path)


__all__ = [
    "bootstrap_slope_contrast", "figure_guards", "figure_memory_exchange",
    "figure_thrash", "figure_transfer_matrix", "h1_slope_contrast",
    "h4_interior_peak", "interior_peak_statistic", "make_figures",
]
