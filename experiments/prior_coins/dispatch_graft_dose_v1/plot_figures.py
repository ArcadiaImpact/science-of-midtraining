"""Dose curves for graft-dose v1, from a collated ``summary.json``. CPU-only.

Three figures:

1. ``dose_response_<slice>.pdf`` — cross-arm directional separation vs unique
   task dose, one line per endpoint, with the two extension cells marked.
   The pre-AFT line is the PRIMARY readout (no AFT seed noise).
2. ``mixture_dose_<slice>.pdf`` — separation vs dose, one line per AFT
   mixture, panel per step. Does the 2% conflict-label override threshold move
   with dose?
3. ``presentations.pdf`` — the 2x2: d8m_x1 vs d8m (presentations at fixed
   unique data) and d2m_x16 vs d8m (unique data at matched compute).

Every point carries its n; the control is drawn as a raw-rate band and is never
a separation partner.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CONFLICT_SLICES = ("eval_trained_conflict", "eval_holdout_conflict")


def wilson_halfwidth(rate: float, n: int, z: float = 1.96) -> float:
    if not n:
        return 0.0
    denominator = 1 + z * z / n
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    return margin / denominator


def separation_halfwidth(row: dict[str, Any]) -> float:
    """Propagate binomial variance across the four rates in the separation."""

    total = 0.0
    for rates, n in (
        (row["charter_rates"], row["charter_n"]),
        (row["coin_rates"], row["coin_n"]),
    ):
        for side in ("charter", "coin"):
            total += wilson_halfwidth(float(rates.get(side, 0.0)), int(n)) ** 2
    return math.sqrt(total)


def base_rows(summary: dict[str, Any], slice_name: str) -> list[dict[str, Any]]:
    return [
        row
        for row in summary["separations"]
        if row["slice"] == slice_name and row["presentations"] == 4
    ]


def plot_dose_response(summary: dict[str, Any], slice_name: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = base_rows(summary, slice_name)
    if not rows:
        return
    endpoints = sorted({row["endpoint"] for row in rows}, key=_endpoint_sort)
    figure, axes = plt.subplots(figsize=(7.5, 5))
    for endpoint in endpoints:
        series = sorted(
            (r for r in rows if r["endpoint"] == endpoint), key=lambda r: r["dose_m"]
        )
        values = [r["separation"] for r in series]
        if any(v is None for v in values):
            continue
        primary = endpoint == "pre_aft"
        axes.errorbar(
            [r["dose_m"] for r in series],
            values,
            yerr=[separation_halfwidth(r) for r in series],
            marker="o",
            linewidth=2.4 if primary else 1.2,
            alpha=1.0 if primary else 0.65,
            capsize=2,
            label=("pre-AFT (graft only)" if primary else endpoint),
            zorder=3 if primary else 2,
        )
    axes.axhline(0, color="0.6", linewidth=0.8, zorder=1)
    axes.set_xscale("log", base=2)
    axes.set_xticks([r["dose_m"] for r in rows])
    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_xlabel("unique task tokens per arm (M)")
    axes.set_ylabel("cross-arm directional separation")
    axes.set_title(
        f"Grafted prior vs dose — {slice_name.replace('eval_', '').replace('_', ' ')}\n"
        f"n = {rows[0]['charter_n']}/{rows[0]['coin_n']} conflict runs per point"
    )
    axes.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(out / f"dose_response_{slice_name}.pdf")
    plt.close(figure)


def _endpoint_sort(name: str) -> tuple[int, str, int]:
    if name == "pre_aft":
        return (0, "", 0)
    mixture, _, step = name.rpartition("_step")
    return (1, mixture, int(step) if step.isdigit() else 0)


def plot_mixture_dose(summary: dict[str, Any], slice_name: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = base_rows(summary, slice_name)
    steps = sorted(
        {
            int(r["endpoint"].rpartition("_step")[2])
            for r in rows
            if r["endpoint"] != "pre_aft"
        }
    )
    if not steps:
        return
    figure, axes_list = plt.subplots(
        1, len(steps), figsize=(4.2 * len(steps), 4.4), sharey=True, squeeze=False
    )
    for axes, step in zip(axes_list[0], steps, strict=True):
        mixtures = sorted(
            {
                r["endpoint"].rpartition("_step")[0]
                for r in rows
                if r["endpoint"].endswith(f"_step{step}")
            }
        )
        for mixture in mixtures:
            series = sorted(
                (r for r in rows if r["endpoint"] == f"{mixture}_step{step}"),
                key=lambda r: r["dose_m"],
            )
            values = [r["separation"] for r in series]
            if any(v is None for v in values):
                continue
            axes.errorbar(
                [r["dose_m"] for r in series],
                values,
                yerr=[separation_halfwidth(r) for r in series],
                marker="o",
                capsize=2,
                label=mixture,
            )
        pre = sorted(
            (r for r in rows if r["endpoint"] == "pre_aft"), key=lambda r: r["dose_m"]
        )
        if pre and all(r["separation"] is not None for r in pre):
            axes.plot(
                [r["dose_m"] for r in pre],
                [r["separation"] for r in pre],
                color="0.35",
                linestyle="--",
                linewidth=1.4,
                label="pre-AFT",
            )
        axes.axhline(0, color="0.6", linewidth=0.8)
        axes.set_xscale("log", base=2)
        axes.set_xticks([r["dose_m"] for r in rows])
        axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes.set_xlabel("unique task tokens (M)")
        axes.set_title(f"AFT step {step}")
    axes_list[0][0].set_ylabel("cross-arm directional separation")
    axes_list[0][-1].legend(fontsize=8, loc="best")
    figure.suptitle(
        f"Conflict-label dose vs data dose — {slice_name.replace('eval_', '')}"
    )
    figure.tight_layout()
    figure.savefig(out / f"mixture_dose_{slice_name}.pdf")
    plt.close(figure)


def plot_presentations(summary: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wanted = {(2, 16), (8, 1), (8, 4), (2, 4)}
    rows = [
        row
        for row in summary["separations"]
        if (row["dose_m"], row["presentations"]) in wanted
        and row["slice"] == "eval_holdout_conflict"
    ]
    if not rows:
        return
    endpoints = sorted({r["endpoint"] for r in rows}, key=_endpoint_sort)
    labels = {
        (2, 4): "2M x4",
        (2, 16): "2M x16 (iso-compute)",
        (8, 1): "8M x1",
        (8, 4): "8M x4",
    }
    figure, axes = plt.subplots(figsize=(8, 4.6))
    width = 0.8 / max(len(endpoints), 1)
    keys = [(2, 4), (2, 16), (8, 1), (8, 4)]
    for index, endpoint in enumerate(endpoints):
        values, errors, positions = [], [], []
        for position, key in enumerate(keys):
            match = [
                r
                for r in rows
                if (r["dose_m"], r["presentations"]) == key
                and r["endpoint"] == endpoint
            ]
            if not match or match[0]["separation"] is None:
                continue
            values.append(match[0]["separation"])
            errors.append(separation_halfwidth(match[0]))
            positions.append(position + index * width - 0.4)
        if values:
            axes.bar(
                positions,
                values,
                width=width,
                yerr=errors,
                capsize=2,
                label="pre-AFT" if endpoint == "pre_aft" else endpoint,
            )
    axes.axhline(0, color="0.4", linewidth=0.8)
    axes.set_xticks(range(len(keys)))
    axes.set_xticklabels([labels[k] for k in keys])
    axes.set_ylabel("cross-arm directional separation")
    axes.set_title(
        "Presentations vs unique tokens (held-out conflict)\n"
        "8M x1 vs 8M x4: passes at fixed data. 2M x16 vs 8M x4: data at fixed compute."
    )
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "presentations.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for slice_name in CONFLICT_SLICES:
        plot_dose_response(summary, slice_name, args.out)
        plot_mixture_dose(summary, slice_name, args.out)
    plot_presentations(summary, args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
