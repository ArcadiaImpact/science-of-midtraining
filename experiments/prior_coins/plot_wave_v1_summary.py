"""Colleague-facing summary figures for the wave-v1 prior-coins results.

This script makes one bar-chart-style figure for each of the five headline
claims in the wave-1 summary.  The figures deliberately use the scored run
artifact rather than transcribing values from ``WAVE_V1_RESULTS.md``.

Run from the repository root::

    .venv/bin/python experiments/prior_coins/plot_wave_v1_summary.py

Use ``--results`` or ``--figures`` to read/write elsewhere.  Both PNG and SVG
versions are written so the plots can be shared directly or edited later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from plot_dispatch_v4_aft import GRID, INK, MUTED, style, wilson  # noqa: E402


ENDPOINT = "step512"
MIXTURES = ("agreement", "coin2", "charter2", "mixed_balanced")
MIX_LABEL = {
    "agreement": "100% agreement",
    "coin2": "+2% coin",
    "charter2": "+2% Charter",
    "mixed_balanced": "10% / 10%",
}
# All hues are drawn from the seaborn "colorblind" palette; the light variants
# are those hues blended 50% toward white.
MIX_COLOR = {
    "agreement": "#029e73",
    "coin2": "#de8f05",
    "charter2": "#0173b2",
    "mixed_balanced": "#cc78bc",
}
CELLS = (("real", "1x"), ("real", "4x"), ("fake", "1x"), ("fake", "4x"))
#: display names: the artifacts and scored keys say real/fake, the figures say
#: true/late (docs before instruct training vs inserted after most of it)
LINEAGE_LABEL = {"real": "true", "fake": "late"}
PARENTS = tuple(
    f"{arm}_{lineage}_{dose}"
    for lineage, dose in CELLS
    for arm in ("charter", "coin")
) + ("control_1x", "control_4x")

CHARTER = "#0173b2"
COIN = "#de8f05"
OTHER = "#949494"
MALFORMED = "#22221f"
LIGHT_OUTCOME_COLOR = {
    "charter": "#80b9d8",
    "coin": "#eec782",
    "other": "#c9c9c9",
    "malformed": "#90908f",
}
DOSE_COLOR = {"1x": "#56b4e9", "4x": "#0173b2"}
LINEAGE_COLOR = {"real": "#029e73", "fake": "#d55e00"}


def save_figure(fig, output: Path) -> None:
    """Write a high-resolution PNG and an editable SVG."""
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def heading(fig, number: int, title: str, subtitle: str) -> None:
    fig.suptitle(
        f"Figure {number}. {title}",
        x=0.08,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.08, 0.925, subtitle, ha="left", va="top", color=MUTED, fontsize=10)


def competence(scored: dict, parent: str, mixture: str) -> float:
    key = f"{parent}|{mixture}|{ENDPOINT}|trained"
    return scored["competence"][key]["accuracy"]


def rate(
    scored: dict,
    parent: str,
    mixture: str,
    slice_name: str,
    endpoint: str = ENDPOINT,
) -> dict:
    key = f"{parent}|{mixture}|{endpoint}"
    return scored["rates"][key][slice_name]


def separation(
    scored: dict, lineage: str, dose: str, mixture: str, condition: str
) -> float:
    key = f"{lineage}|{dose}|{mixture}|{ENDPOINT}|{condition}"
    return scored["separation"][key]["separation"]


def annotate_bars(ax, bars, *, fmt="{:.2f}", offset=0.018, fontsize=9) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=INK,
        )


def figure_0(scored: dict, output: Path) -> None:
    """All final cells learn the in-distribution agreement task."""
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    xs = list(range(len(MIXTURES)))
    means = []
    all_values = {}
    for mixture in MIXTURES:
        values = [competence(scored, parent, mixture) * 100 for parent in PARENTS]
        all_values[mixture] = values
        means.append(sum(values) / len(values))

    ax.bar(
        xs,
        means,
        width=0.64,
        color=[MIX_COLOR[m] for m in MIXTURES],
        edgecolor="white",
        linewidth=1.4,
        zorder=2,
    )
    # Eight filled circles are midtrained parents; the two outlined diamonds are
    # the no-document controls.  Fixed offsets make reruns deterministic.
    offsets = (-0.22, -0.16, -0.10, -0.04, 0.04, 0.10, 0.16, 0.22)
    for x, mixture in zip(xs, MIXTURES):
        values = all_values[mixture]
        ax.scatter(
            [x + off for off in offsets],
            values[:8],
            s=24,
            color=INK,
            alpha=0.72,
            zorder=4,
        )
        ax.scatter(
            [x - 0.08, x + 0.08],
            values[8:],
            s=42,
            marker="D",
            facecolor="white",
            edgecolor=INK,
            linewidth=1.2,
            zorder=5,
        )
        ax.text(x, means[x] - 4.2, f"{means[x]:.1f}%", ha="center", color="white",
                fontsize=10, fontweight="bold", zorder=5)

    ax.set_xticks(xs)
    ax.set_xticklabels([MIX_LABEL[m] for m in MIXTURES])
    ax.set_ylim(0, 105)
    ax.axhline(100, color=GRID, linewidth=1.0, zorder=1)
    style(ax, ylabel="trained-clause agreement accuracy (%)")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=INK,
                   label="midtrained parent"),
            Line2D([], [], marker="D", linestyle="none", markerfacecolor="white",
                   markeredgecolor=INK, color=INK, label="no-document control"),
        ],
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncol=2,
    )
    fig.suptitle(
        "Figure 0",
        x=0.08,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.92, bottom=0.16)
    fig.text(
        0.98,
        0.015,
        "Mean across 10 parents. Dots: individual parents. Controls saw no "
        "Charter/coin documents.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    save_figure(fig, output / "figure_0_id_task_accuracy")


def _stacked_choice_bar(ax, y: float, counts: dict, n: int) -> None:
    """Draw one multinomial composition bar with boundary uncertainty.

    A category's uncertainty cannot be centred on its location inside a stacked
    bar: the location depends on every category to its left.  Instead, whiskers
    show Wilson 95% intervals for each internal cumulative boundary.  Those
    boundaries fully determine the four segment widths and preserve the fact
    that the outcomes sum to 100%.
    """
    groups = (
        (counts.get("charter", 0), CHARTER, "Charter"),
        (counts.get("coin", 0), COIN, "coin"),
        (counts.get("other", 0), OTHER, "other crew"),
        (counts.get("malformed", 0), MALFORMED, "malformed"),
    )
    left = 0.0
    cumulative = 0
    for index, (count, color, _) in enumerate(groups):
        width = count / n * 100 if n else 0.0
        ax.barh(y, width, left=left, height=0.72, color=color, edgecolor="white",
                linewidth=1.2, zorder=3)
        if width >= 6:
            ax.text(left + width / 2, y, f"{width:.0f}", ha="center", va="center",
                    color=INK if color == OTHER else "white", fontsize=8.5, zorder=4)
        left += width
        cumulative += count
        if index == len(groups) - 1 or not n:
            continue
        p = cumulative / n
        lo, hi = wilson(p, n)
        centre = p * 100
        xerr = [[lo * 100], [hi * 100]]
        # A white underlay keeps the whisker legible on both dark and light
        # segments; the thin ink line is the actual interval.
        ax.errorbar(
            [centre], [y], xerr=xerr, fmt="none", ecolor="white",
            elinewidth=3.0, capsize=4.0, capthick=3.0, zorder=5,
        )
        ax.errorbar(
            [centre], [y], xerr=xerr, fmt="none", ecolor=INK,
            elinewidth=1.0, capsize=3.0, capthick=1.0, zorder=6,
        )


def figure_1(scored: dict, output: Path) -> None:
    """Held-out conflict episodes, restricted to Charter-held-in clauses."""
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    rows = []
    y = 0.0
    pair_centres = []
    for lineage, dose in CELLS:
        pair = []
        for arm in ("charter", "coin"):
            parent = f"{arm}_{lineage}_{dose}"
            block = rate(scored, parent, "agreement", "eval_trained_conflict")
            _stacked_choice_bar(ax, y, block["counts"], block["n"])
            rows.append((y, f"{LINEAGE_LABEL[lineage]} {dose} · {arm} prior"))
            pair.append(y)
            y += 1.0
        pair_centres.append((sum(pair) / 2, lineage, dose))
        y += 0.42

    control_start = y
    for dose in ("1x", "4x"):
        block = rate(scored, f"control_{dose}", "agreement", "eval_trained_conflict")
        _stacked_choice_bar(ax, y, block["counts"], block["n"])
        rows.append((y, f"control {dose} · no documents"))
        y += 1.0
    ax.axhline(control_start - 0.35, color=GRID, linewidth=1.4, zorder=2)

    for centre, lineage, dose in pair_centres:
        sep = separation(scored, lineage, dose, "agreement", "trained")
        ax.text(102, centre, f"sep. {sep:+.2f}", va="center", ha="left",
                color=INK, fontsize=9, fontweight="bold")

    ax.set_yticks([row[0] for row in rows])
    ax.set_yticklabels([row[1] for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 113)
    ax.set_xlabel("share of held-out conflict-eval runs (%)", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(
        handles=[
            Patch(facecolor=CHARTER, label="chose Charter"),
            Patch(facecolor=COIN, label="chose coin / cheapest"),
            Patch(facecolor=OTHER, label="chose another crew"),
            Patch(facecolor=MALFORMED, label="malformed answer"),
        ],
        frameon=False,
        fontsize=9,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
    )
    heading(
        fig,
        1,
        "The midtraining direction generalizes to held-out motivational conflict",
        "All episodes are held out; this view uses the five Charter clauses held in "
        "during AFT. Training uses only prior-neutral agreement examples.",
    )
    fig.subplots_adjust(top=0.84, left=0.24, bottom=0.16)
    fig.text(
        0.98,
        0.015,
        "Whiskers: Wilson 95% intervals on cumulative category boundaries. The "
        "no-document controls are shown as raw rates, never as separation partners.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    save_figure(fig, output / "figure_1_ood_directional_generalisation")


def figure_1_minibars(scored: dict, output: Path) -> None:
    """Alternative Figure 1 with direct, non-cumulative category intervals.

    Unlike the stacked version, every outcome bar starts at zero.  Its whisker
    is therefore the Wilson interval for that outcome's own binomial rate, not
    for a cumulative boundary.  Kept as a separate output while the two designs
    are being compared.
    """
    fig, ax = plt.subplots(figsize=(11.2, 6.7))
    categories = (
        ("charter", CHARTER, "chose Charter"),
        ("coin", COIN, "chose coin / cheapest"),
        ("other", OTHER, "chose another crew"),
        ("malformed", MALFORMED, "malformed answer"),
    )
    offsets = (-0.30, -0.10, 0.10, 0.30)
    bar_height = 0.18
    rows: list[tuple[float, str]] = []
    y = 0.0

    def draw_parent(parent: str, centre: float, endpoint: str) -> None:
        block = rate(
            scored,
            parent,
            "agreement",
            "eval_trained_conflict",
            endpoint,
        )
        counts, n = block["counts"], block["n"]
        for offset, (verdict, color, _) in zip(offsets, categories):
            value = counts.get(verdict, 0) / n if n else 0.0
            lo, hi = wilson(value, n)
            yy = centre + offset
            ax.barh(
                yy,
                value * 100,
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.errorbar(
                [value * 100],
                [yy],
                xerr=[[lo * 100], [hi * 100]],
                fmt="none",
                ecolor=INK,
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                zorder=5,
            )
            label_x = value * 50 if value >= 0.09 else (value + hi) * 100 + 0.8
            label_color = INK if value < 0.09 else "white"
            ax.text(
                label_x,
                yy,
                f"{value * 100:.0f}",
                ha="center" if value >= 0.09 else "left",
                va="center",
                color=label_color,
                fontsize=7.7,
                zorder=6,
            )

    for lineage in ("real",):
        for arm_index, arm in enumerate(("charter", "coin")):
            if arm_index:
                # the same light dashed group rule Figures 2-3 use
                y += 0.35
                ax.axhline(y - 0.48, color=GRID, linewidth=0.9,
                           linestyle=(0, (4, 3)), zorder=2)
            parent = f"{arm}_{lineage}_4x"
            for endpoint, stage in (("baseline", "pre-AFT"),
                                    (ENDPOINT, "post-AFT")):
                draw_parent(parent, y, endpoint)
                rows.append((y, f"{LINEAGE_LABEL[lineage]} 4x · {arm} prior "
                                f"· {stage}"))
                y += 1.05
        y += 0.45

    control_start = y
    for endpoint, stage in (("baseline", "pre-AFT"),
                            (ENDPOINT, "post-AFT")):
        draw_parent("control_4x", y, endpoint)
        rows.append((y, f"control 4x · no documents · {stage}"))
        y += 1.05
    ax.axhline(control_start - 0.48, color=GRID, linewidth=1.4, zorder=2)

    ax.set_yticks([row[0] for row in rows])
    ax.set_yticklabels([row[1] for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of held-out conflict-eval runs (%)", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(
        handles=[Patch(facecolor=color, label=label)
                 for _, color, label in categories],
        frameon=False,
        fontsize=9,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.suptitle(
        "Figure 1",
        x=0.08,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.92, left=0.24, bottom=0.17)
    save_figure(fig, output / "figure_1_ood_directional_generalisation_minibars")


def _comparison_minibars(
    scored: dict,
    output: Path,
    *,
    number: int,
    groups: tuple[tuple[tuple[str, ...], ...], ...],
    condition: str,
    output_name: str,
    colors: dict[str, str] | None = None,
    control_group: int | None = None,
    group_separators: bool = False,
) -> None:
    """Shared mini-bar layout for Figures 2–5.

    Each row specification is ``(parent, mixture, display_label)`` for a
    post-AFT row, or ``(parent, mixture, endpoint, display_label)`` when the
    checkpoint should be explicit. Groups only control row spacing — plus,
    with ``group_separators``, a light dashed rule between adjacent groups
    (the control boundary keeps its heavier solid rule). The underlying bars,
    category order, scales and Wilson intervals are identical across figures.
    """
    palette = colors or {
        "charter": CHARTER,
        "coin": COIN,
        "other": OTHER,
        "malformed": MALFORMED,
    }
    categories = (
        ("charter", "chose Charter"),
        ("coin", "chose coin / cheapest"),
        ("other", "chose another crew"),
        ("malformed", "malformed answer"),
    )
    offsets = (-0.30, -0.10, 0.10, 0.30)
    bar_height = 0.18
    n_rows = sum(len(group) for group in groups)
    fig_height = max(6.8, 0.63 * n_rows + 2.25)
    fig, ax = plt.subplots(figsize=(11.2, fig_height))
    rows: list[tuple[float, str]] = []
    y = 0.0

    for group_index, group in enumerate(groups):
        if group_index == control_group:
            ax.axhline(y - 0.48, color=GRID, linewidth=1.4, zorder=2)
        elif group_separators and group_index:
            ax.axhline(y - 0.48, color=GRID, linewidth=0.9,
                       linestyle=(0, (4, 3)), zorder=2)
        for row in group:
            if len(row) == 3:
                parent, mixture, label = row
                endpoint = ENDPOINT
            else:
                parent, mixture, endpoint, label = row
            slice_name = ("eval_trained_conflict" if condition == "trained"
                          else "eval_holdout_conflict")
            block = rate(scored, parent, mixture, slice_name, endpoint)
            counts, n = block["counts"], block["n"]
            for offset, (verdict, _) in zip(offsets, categories):
                value = counts.get(verdict, 0) / n if n else 0.0
                lo, hi = wilson(value, n)
                yy = y + offset
                color = palette[verdict]
                ax.barh(
                    yy,
                    value * 100,
                    height=bar_height,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
                ax.errorbar(
                    [value * 100],
                    [yy],
                    xerr=[[lo * 100], [hi * 100]],
                    fmt="none",
                    ecolor=INK,
                    elinewidth=1.0,
                    capsize=2.5,
                    capthick=1.0,
                    zorder=5,
                )
                label_x = value * 50 if value >= 0.09 else (value + hi) * 100 + 0.8
                use_dark_text = colors is not None or value < 0.09
                ax.text(
                    label_x,
                    yy,
                    f"{value * 100:.0f}",
                    ha="center" if value >= 0.09 else "left",
                    va="center",
                    color=INK if use_dark_text else "white",
                    fontsize=7.7,
                    zorder=6,
                )
            rows.append((y, label))
            y += 1.05
        y += 0.45

    ax.set_yticks([row[0] for row in rows])
    ax.set_yticklabels([row[1] for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of held-out conflict-eval runs (%)", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    # a short figure needs the legend pushed further below the xlabel: the
    # anchor offset is a fraction of a smaller axes height
    short = n_rows < 8
    ax.legend(
        handles=[Patch(facecolor=palette[verdict], label=label)
                 for verdict, label in categories],
        frameon=False,
        fontsize=9,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18 if short else -0.12),
    )
    fig.suptitle(
        f"Figure {number}",
        x=0.08,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.92, left=0.29, bottom=0.17 if short else 0.13)
    save_figure(fig, output / output_name)


def figure_2_minibars(scored: dict, output: Path) -> None:
    """Dose comparison, grouped so each 1x/4x contrast is adjacent.

    The no-document controls are shown as their own group below the rule —
    unpaired, and remember their 1x/4x is a replay-dose contrast, not an
    arm-dose one.
    """
    groups = tuple(
        tuple(
            (f"{arm}_{lineage}_{dose}", "agreement",
             f"{LINEAGE_LABEL[lineage]} · {arm} prior · {dose}")
            for dose in ("1x", "4x")
        )
        for lineage in ("real", "fake")
        for arm in ("charter", "coin")
    )
    controls = (
        ("control_1x", "agreement", "control · no documents · 1x"),
        ("control_4x", "agreement", "control · no documents · 4x"),
    )
    _comparison_minibars(
        scored,
        output,
        number=2,
        groups=groups + (controls,),
        condition="trained",
        output_name="figure_2_higher_dose_generalisation_minibars",
        control_group=len(groups),
        group_separators=True,
    )


def figure_3_minibars(scored: dict, output: Path) -> None:
    """Pipeline-position comparison at 4x, grouped so real/fake is adjacent."""
    groups = tuple(
        tuple(
            (f"{arm}_{lineage}_4x", "agreement",
             f"4x · {arm} prior · {LINEAGE_LABEL[lineage]}")
            for lineage in ("real", "fake")
        )
        for arm in ("charter", "coin")
    )
    controls = (
        ("control_4x", "agreement", "4x · control · no documents"),
    )
    _comparison_minibars(
        scored,
        output,
        number=3,
        groups=groups + (controls,),
        condition="trained",
        output_name="figure_3_real_vs_fake_midtraining_minibars",
        control_group=len(groups),
        group_separators=True,
    )


def figure_4_minibars(scored: dict, output: Path) -> None:
    """Every prior is paired with the 2% label direction that opposes it."""
    arm_groups = tuple(
        (
            (f"charter_{lineage}_{dose}", "coin2",
             f"{LINEAGE_LABEL[lineage]} {dose} · Charter prior · +2% coin labels"),
            (f"coin_{lineage}_{dose}", "charter2",
             f"{LINEAGE_LABEL[lineage]} {dose} · coin prior · +2% Charter labels"),
        )
        for lineage, dose in CELLS
    )
    controls = (
        ("control_1x", "coin2", "control 1x · +2% coin labels"),
        ("control_1x", "charter2", "control 1x · +2% Charter labels"),
        ("control_4x", "coin2", "control 4x · +2% coin labels"),
        ("control_4x", "charter2", "control 4x · +2% Charter labels"),
    )
    _comparison_minibars(
        scored,
        output,
        number=4,
        groups=arm_groups + (controls,),
        condition="trained",
        output_name="figure_4_conflict_overwrites_prior_minibars",
        control_group=len(arm_groups),
    )


def figure_4_4x_minibars(scored: dict, output: Path) -> None:
    """A separate 4x-only variant of Figure 4 for side-by-side comparison.

    Row order is lineage, then prior, then label direction — Charter labels
    always before coin labels, so the two rows of every group read the same
    way rather than opposite-prior-first.
    """
    arm_groups = tuple(
        rows
        for lineage in ("real", "fake")
        for rows in (
            (
                (f"charter_{lineage}_4x", "charter2",
                 f"{LINEAGE_LABEL[lineage]} 4x · Charter prior · +2% Charter labels"),
                (f"charter_{lineage}_4x", "coin2",
                 f"{LINEAGE_LABEL[lineage]} 4x · Charter prior · +2% coin labels"),
            ),
            (
                (f"coin_{lineage}_4x", "charter2",
                 f"{LINEAGE_LABEL[lineage]} 4x · coin prior · +2% Charter labels"),
                (f"coin_{lineage}_4x", "coin2",
                 f"{LINEAGE_LABEL[lineage]} 4x · coin prior · +2% coin labels"),
            ),
        )
    )
    controls = (
        ("control_4x", "charter2", "control 4x · +2% Charter labels"),
        ("control_4x", "coin2", "control 4x · +2% coin labels"),
    )
    _comparison_minibars(
        scored,
        output,
        number=4,
        groups=arm_groups + (controls,),
        condition="trained",
        output_name="figure_4_conflict_overwrites_prior_4x_minibars",
        control_group=len(arm_groups),
    )


def figure_5_minibars(scored: dict, output: Path) -> None:
    """The Figure-1 model set, evaluated on Charter-held-out clauses."""
    arm_groups = tuple(
        (
            (f"charter_{lineage}_{dose}", "agreement",
             f"{LINEAGE_LABEL[lineage]} {dose} · charter prior"),
            (f"coin_{lineage}_{dose}", "agreement",
             f"{LINEAGE_LABEL[lineage]} {dose} · coin prior"),
        )
        for lineage, dose in CELLS
    )
    controls = (
        ("control_1x", "agreement", "control 1x · no documents"),
        ("control_4x", "agreement", "control 4x · no documents"),
    )
    _comparison_minibars(
        scored,
        output,
        number=5,
        groups=arm_groups + (controls,),
        condition="holdout",
        output_name="figure_5_unseen_charter_rules_minibars",
        colors=LIGHT_OUTCOME_COLOR,
        control_group=len(arm_groups),
    )


def figure_5_4x_pre_post_minibars(scored: dict, output: Path) -> None:
    """Separate Figure 5 variant: 4x charter-prior arms at pre- and post-AFT.

    The coin-prior arms are deliberately absent: the question here is whether
    the Charter preference reaches clauses AFT never drilled, read against the
    no-document control.
    """
    arm_groups = tuple(
        (
            (f"charter_{lineage}_4x", "agreement", "baseline",
             f"{LINEAGE_LABEL[lineage]} 4x · charter prior · pre-AFT"),
            (f"charter_{lineage}_4x", "agreement", ENDPOINT,
             f"{LINEAGE_LABEL[lineage]} 4x · charter prior · post-AFT"),
        )
        for lineage in ("real", "fake")
    )
    controls = (
        ("control_4x", "agreement", "baseline",
         "control 4x · no documents · pre-AFT"),
        ("control_4x", "agreement", ENDPOINT,
         "control 4x · no documents · post-AFT"),
    )
    _comparison_minibars(
        scored,
        output,
        number=5,
        groups=arm_groups + (controls,),
        condition="holdout",
        output_name="figure_5_unseen_charter_rules_4x_pre_post_minibars",
        colors=LIGHT_OUTCOME_COLOR,
        control_group=len(arm_groups),
    )


def figure_2(scored: dict, output: Path) -> None:
    """Within each lineage, compare held-out separation at 1x and 4x dose."""
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    x = [0, 1]
    width = 0.32
    for offset, dose in ((-width / 2, "1x"), (width / 2, "4x")):
        values = [separation(scored, lineage, dose, "agreement", "holdout")
                  for lineage in ("real", "fake")]
        bars = ax.bar([v + offset for v in x], values, width, color=DOSE_COLOR[dose],
                      edgecolor="white", linewidth=1.2, label=dose, zorder=3)
        annotate_bars(ax, bars, offset=0.018)
    for i, lineage in enumerate(("real", "fake")):
        one = separation(scored, lineage, "1x", "agreement", "holdout")
        four = separation(scored, lineage, "4x", "agreement", "holdout")
        ax.text(i, max(one, four) + 0.095, f"Δ {four - one:+.2f}", ha="center",
                color=MUTED, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["true midtraining", "late midtraining"])
    ax.set_ylim(0, 0.82)
    ax.axhline(0, color=INK, linewidth=1.0)
    style(ax, ylabel="OOD directional separation (0–2)")
    ax.legend(frameon=False, fontsize=9, title="midtraining dose", title_fontsize=9)
    heading(
        fig,
        2,
        "A higher midtraining dose improves generalization",
        "The 4× parent has higher held-out separation within both pipeline "
        "lineages, but the gains are modest (+0.19 and +0.13).",
    )
    fig.subplots_adjust(top=0.82, bottom=0.16)
    fig.text(
        0.98,
        0.015,
        "Dose is only compared within lineage: real 1×→4× and fake 1×→4× use "
        "different training-dose definitions. No-doc controls are unpaired.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    save_figure(fig, output / "figure_2_higher_dose_generalisation")


def figure_3(scored: dict, output: Path) -> None:
    """At each dose, compare real and fake pipeline position."""
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    x = [0, 1]
    width = 0.32
    for offset, lineage in ((-width / 2, "real"), (width / 2, "fake")):
        values = [separation(scored, lineage, dose, "agreement", "holdout")
                  for dose in ("1x", "4x")]
        bars = ax.bar(
            [v + offset for v in x],
            values,
            width,
            color=LINEAGE_COLOR[lineage],
            edgecolor="white",
            linewidth=1.2,
            label=LINEAGE_LABEL[lineage],
            zorder=3,
        )
        annotate_bars(ax, bars, offset=0.018)
    for i, dose in enumerate(("1x", "4x")):
        real = separation(scored, "real", dose, "agreement", "holdout")
        fake = separation(scored, "fake", dose, "agreement", "holdout")
        ax.text(i, max(real, fake) + 0.095, f"Δ {real - fake:+.2f}", ha="center",
                color=MUTED, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["1× dose", "4× dose"])
    ax.set_ylim(0, 0.82)
    ax.axhline(0, color=INK, linewidth=1.0)
    style(ax, ylabel="OOD directional separation (0–2)")
    ax.legend(frameon=False, fontsize=9, title="pipeline position",
              title_fontsize=9)
    heading(
        fig,
        3,
        "Real midtraining generalizes somewhat better than fake midtraining",
        "Putting the documents before instruct training beats post-instruct SDF at "
        "both doses, while both pipeline positions retain an OOD effect.",
    )
    fig.subplots_adjust(top=0.82, bottom=0.15)
    fig.text(
        0.98,
        0.015,
        "Real: arm documents before instruct training. Fake: SDF on arm documents "
        "after 90% of instruct training, followed by the final 10%.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    save_figure(fig, output / "figure_3_real_vs_fake_midtraining")


def figure_4(scored: dict, output: Path) -> None:
    """Compare prior-neutral AFT with the two one-directional 2% mixtures."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), sharey=True)
    width = 0.23
    for ax, lineage in zip(axes, ("real", "fake")):
        x = [0, 1]
        for index, mixture in enumerate(("agreement", "coin2", "charter2")):
            offset = (index - 1) * width
            values = [separation(scored, lineage, dose, mixture, "trained")
                      for dose in ("1x", "4x")]
            bars = ax.bar(
                [v + offset for v in x],
                values,
                width,
                color=MIX_COLOR[mixture],
                edgecolor="white",
                linewidth=1.1,
                label=MIX_LABEL[mixture],
                zorder=3,
            )
            annotate_bars(ax, bars, offset=0.025, fontsize=8.5)
        residuals = {
            dose: sum(
                separation(scored, lineage, dose, mixture, "trained")
                for mixture in ("coin2", "charter2")
            ) / 2
            for dose in ("1x", "4x")
        }
        ax.text(
            0.03,
            0.95,
            f"mean 2% residual: {residuals['1x']:.2f} → {residuals['4x']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=MUTED,
            fontsize=9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["1× dose", "4× dose"])
        ax.set_title(f"{LINEAGE_LABEL[lineage]} midtraining", color=INK,
                     fontsize=11, loc="left")
        ax.axhline(0, color=INK, linewidth=1.0, zorder=2)
        style(ax, ylabel="trained-clause directional separation (0–2)"
              if lineage == "real" else None)
    axes[0].set_ylim(0, 1.68)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    heading(
        fig,
        4,
        "Two percent conflict data is enough to mostly overwrite the prior",
        "Either one-directional 2% mixture collapses the converged readout; the "
        "remaining separation is small and averages slightly higher at 4×.",
    )
    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.15)
    fig.text(
        0.98,
        0.015,
        "Agreement is the prior-neutral reference. The two 2% directions are shown "
        "separately because they are not mechanistically interchangeable. "
        "No-doc controls have no separation partner.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    save_figure(fig, output / "figure_4_conflict_overwrites_prior")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=EXP / "runs" / "dispatch_wave_v1" / "results" / "scored.json",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=EXP / "figures" / "wave_v1_summary",
    )
    parser.add_argument(
        "--minibars-only",
        action="store_true",
        help="render Figures 1–5 in the mini-bar layout without touching the "
             "original Figures 0–4",
    )
    args = parser.parse_args()
    scored = json.loads(args.results.read_text())

    if args.minibars_only:
        for make in (
            figure_1_minibars,
            figure_2_minibars,
            figure_3_minibars,
            figure_4_minibars,
            figure_5_minibars,
            figure_5_4x_pre_post_minibars,
        ):
            make(scored, args.figures)
        return

    figure_0(scored, args.figures)
    figure_1(scored, args.figures)
    figure_2(scored, args.figures)
    figure_3(scored, args.figures)
    figure_4(scored, args.figures)


if __name__ == "__main__":
    main()
