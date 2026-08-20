"""Figure-0-format plots for the template-diversity run.

One figure per condition in {trained, held-out charter clauses} x
{trained, held-out presentation templates} — four figures. Each follows the
classic ``figure_0_ambiguous_vs_unambiguous`` layout from
``plot_wave_v1_summary``: two side-by-side 100%-stacked panels over the same
six rows (charter prior, control, coin prior; grouped by AFT condition),
agreement ("ambiguous") episodes on the left, conflict ("diagnostic") on the
right. All episodes are held out of training in every
panel; the condition names the *clause* and *template* splits.

Numbers come from ``results/scored.json`` (strict parsing — the primary
readout). The held-out-template figures carry a footnote about the T051
telegraph-STOP parse artifact, which dominates their malformed mass; the
lenient numbers live in ``scored.json['lenient_heldout']``.

Run from this directory (matplotlib required; ``uv run --with matplotlib``)::

    python3 plot_figure0_template_diversity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch, Patch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for p in (str(EXP), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from plot_dispatch_v4_aft import GRID, INK, MUTED  # noqa: E402

# palette identical to plot_wave_v1_summary
CHARTER = "#0173b2"
COIN = "#de8f05"
OTHER = "#949494"
MALFORMED = "#22221f"
SHARED = "#029e73"
SEGMENT_ORDER = ("charter", "other", "malformed", "coin")
AGREEMENT_SEGMENT_ORDER = ("shared", "other", "malformed")
CONFLICT_COLOR = {"charter": CHARTER, "coin": COIN, "other": OTHER,
                  "malformed": MALFORMED}
AGREEMENT_COLOR = {"shared": SHARED, "other": OTHER, "malformed": MALFORMED}
CONFLICT_LABEL = {
    "charter": "chose Charter",
    "coin": "chose coin / cheapest",
    "other": "chose another crew",
    "malformed": "malformed answer",
}
AGREEMENT_LABEL = {
    "shared": "chose the (single) correct crew",
    "other": CONFLICT_LABEL["other"],
    "malformed": CONFLICT_LABEL["malformed"],
}

#: (arm key in scored.json, display name, tick-label colour).
#:
#: Ordered charter / control / coin so the matched-dose control sits between
#: the two arms it is the midpoint of. The arm labels carry their own bar
#: colour -- taken from the palette above, so a label cannot drift from the
#: segment it names -- and the control keeps the muted default, having no
#: segment of its own to match.
ROWS = (
    ("charter_real_4x", "charter prior", CHARTER),
    ("gate2_dolmino_4x", "control", None),
    ("coin_real_4x", "coin prior", COIN),
)
#: Grouped coarsely by AFT condition and finely by midtrain arm: within one AFT
#: condition, what did each prior do. The condition names the brace in the left
#: margin, so a row label is just its substrate instead of repeating "pre AFT"
#: three times and "post AFT" three.
ENDPOINTS = (("baseline", "pre AFT"), ("step512", "post AFT"))

#: extra blank row-heights between the two AFT blocks
GROUP_GAP = 0.9

#: the four requested conditions: (clause split, template mode)
CONDITIONS = (
    ("trained", "trained"),
    ("trained", "heldout"),
    ("holdout", "trained"),
    ("holdout", "heldout"),
)
CLAUSE_WORD = {"trained": "trained clauses", "holdout": "held-out clauses"}
MODE_WORD = {"trained": "trained templates", "heldout": "held-out templates"}


def left_of_ticklabels(ax, gap: float = 0.014) -> float:
    """Axes-fraction x just clear of the widest y tick label.

    Measured off the rendered text rather than hardcoded: an offset tuned until
    it clears today's labels silently overlaps them the next time a row is
    renamed, and the overlap only shows up in the PNG. Needs a renderer, so it
    draws the canvas first -- cheap on Agg, and savefig redraws anyway.
    """
    fig = ax.figure
    fig.canvas.draw()
    labels = [lb for lb in ax.get_yticklabels() if lb.get_text()]
    if not labels:
        return -gap
    renderer = fig.canvas.get_renderer()
    x0 = min(lb.get_window_extent(renderer).x0 for lb in labels)
    return ax.transAxes.inverted().transform((x0, 0))[0] - gap


def brace(ax, y0: float, y1: float, label: str, *, x: float | None = None,
          width: float = 0.012, pad: float = 0.008,
          color: str = MUTED, fontsize: float = 9.5) -> None:
    """A curly brace spanning data rows ``y0``..``y1``, left of the axes.

    ``x``/``width``/``pad`` are axes fractions and negative x means "out in the
    left margin, beyond the tick labels"; ``y0``/``y1`` are data coordinates, so
    a brace tracks its rows rather than a fixed pixel offset. ``x=None``
    measures a clear position via :func:`left_of_ticklabels`.

    ``clip_on=False`` because the whole point is to sit outside the axes --
    savefig uses ``bbox_inches="tight"``, so the brace and its label expand the
    saved figure instead of being cropped off it.
    """
    if x is None:
        x = left_of_ticklabels(ax)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    tip, spine = x, x - width          # tips toward the bars, point away
    ctrl, mid = x - width / 2, (y0 + y1) / 2
    q = (y1 - y0) / 4
    verts = [(tip, y0),
             (ctrl, y0), (ctrl, y0 + q),        # lower S, tip up to the waist
             (ctrl, mid), (spine, mid),         # waist, out to the point
             (ctrl, mid), (ctrl, y1 - q),       # upper S, back off the point
             (ctrl, y1), (tip, y1)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 8
    ax.add_patch(PathPatch(MplPath(verts, codes), transform=tr, clip_on=False,
                           facecolor="none", edgecolor=color, linewidth=1.1,
                           joinstyle="round", zorder=5))
    ax.text(spine - pad, mid, label, transform=tr, ha="right", va="center",
            fontsize=fontsize, color=color, clip_on=False, zorder=5)


def block(scored: dict, arm: str, endpoint: str, clauses: str, kind: str,
          mode: str) -> dict:
    """The run-level rates block for one row of one panel."""
    slice_name = f"eval_{clauses}_{kind}__{mode}"
    cell = scored["arms"][arm][endpoint][slice_name]
    return cell[f"{kind}_runs"]


def episode_count(manifest: dict, clauses: str, kind: str, mode: str) -> int:
    return manifest["eval_slices"][f"eval_{clauses}_{kind}__{mode}"]["n"]


def draw_panel(ax, scored: dict, *, clauses: str, kind: str, mode: str,
               order, palette):
    """Six stacked rows; returns the (asserted-common) run count and the
    ``(y0, y1)`` span of each AFT block, so a caller can brace them."""
    ns = set()
    y = 0.0
    ticks, labels, spans = [], [], []
    for group_index, (endpoint, _stage) in enumerate(ENDPOINTS):
        if group_index:
            # the rule goes midway between the blocks it divides: the previous
            # block's last row is at y - 1 and the next starts at y + gap
            ax.axhline(y + (GROUP_GAP - 1.0) / 2, color=GRID, linewidth=1.4,
                       zorder=2)
            y += GROUP_GAP
        first = y
        for arm, arm_label, _color in ROWS:
            b = block(scored, arm, endpoint, clauses, kind, mode)
            n, rates = b["n"], b["rates"]
            ns.add(n)
            left = 0.0
            for verdict in order:
                width = rates.get(verdict, 0.0) * 100
                color = palette[verdict]
                ax.barh(y, width, left=left, height=0.62, color=color,
                        edgecolor="white", linewidth=1.2, zorder=3)
                if width >= 4.5:
                    ax.text(left + width / 2, y, f"{width:.0f}",
                            ha="center", va="center", fontsize=8.4, zorder=4,
                            color=INK if color == OTHER else "white")
                left += width
            ticks.append(y)
            labels.append(arm_label)
            y += 1.0
        spans.append((first, y - 1.0))
    if len(ns) != 1:
        raise ValueError(f"panel rows have differing n {sorted(ns)}; the "
                         "single-n footnote no longer holds")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    # descending limits invert the axis, and are idempotent under sharey --
    # unlike invert_yaxis(), which cancels itself out when both panels call it
    ax.set_ylim(y - 0.5, -0.5)
    return ns.pop(), spans


def figure_0_condition(scored: dict, manifest: dict, output: Path,
                       clauses: str, mode: str) -> None:
    # The parenthetical names the *template* split, which is what varies across
    # the four figures; MODE_WORD keeps it honest, so the trained-template pair
    # says "trained templates" rather than inheriting the held-out wording. That
    # every episode is held out of training is said once, in the footnote.
    panels = (
        (f"Ambiguous ({MODE_WORD[mode]})", "agreement",
         AGREEMENT_SEGMENT_ORDER, AGREEMENT_COLOR, AGREEMENT_LABEL),
        (f"Diagnostic ({MODE_WORD[mode]})", "conflict",
         SEGMENT_ORDER, CONFLICT_COLOR, CONFLICT_LABEL),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.6), sharey=True)
    ns, eps = {}, {}
    for panel_index, (ax, (title, kind, order, palette, labels)) in enumerate(
            zip(axes, panels)):
        ns[kind], spans = draw_panel(ax, scored, clauses=clauses, kind=kind,
                                     mode=mode, order=order, palette=palette)
        eps[kind] = episode_count(manifest, clauses, kind, mode)
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"share of {kind}-eval runs (%)", color=INK, fontsize=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, left=False)
        # after tick_params, which sets every label to MUTED and would
        # otherwise clobber the per-arm colours
        for tick, (_arm, _label, color) in zip(ax.get_yticklabels(), ROWS * 2):
            if color:
                tick.set_color(color)
        # braces on the leftmost panel only -- sharey hides the other's labels
        if panel_index == 0:
            x = left_of_ticklabels(ax)          # measure once for both braces
            for (y0, y1), (_endpoint, stage) in zip(spans, ENDPOINTS):
                brace(ax, y0, y1, stage, x=x)
        ax.legend(
            handles=[Patch(facecolor=palette[v], label=labels[v])
                     for v in order],
            frameon=False, fontsize=9, ncol=len(order),
            loc="upper center", bbox_to_anchor=(0.5, -0.13),
        )

    fig.suptitle(
        f"Figure 0 — {CLAUSE_WORD[clauses]} × {MODE_WORD[mode]}",
        x=0.055, y=0.985, ha="left", color=INK, fontsize=14, fontweight="bold",
    )
    note = (
        f"All episodes held out of training. {CLAUSE_WORD[clauses]}, "
        f"{MODE_WORD[mode]}; n = {eps['agreement']:,} episodes "
        f"({ns['agreement']:,} runs) per ambiguous row and "
        f"{eps['conflict']:,} episodes ({ns['conflict']:,} conflict runs) "
        "per diagnostic row."
    )
    if mode == "heldout":
        note += (
            "  Strict parsing: the malformed mass here is dominated by the "
            "T051 telegraph in-voice 'STOP' parse artifact (lenient: ~1%; "
            "see scored.json lenient_heldout)."
        )
    fig.text(0.985, 0.015, note, ha="right", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(top=0.84, bottom=0.24, left=0.155, right=0.985,
                        wspace=0.08)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"figure_0_{clauses}_clauses_{mode}_templates"
    for suffix in (".png", ".svg"):
        path = (output / stem).with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=EXP / "runs" / "template_diversity_v1" / "results" / "scored.json",
    )
    parser.add_argument(
        "--data-manifest", type=Path,
        default=EXP / "runs" / "template_diversity_v1" / "data" / "dataset_manifest.json",
    )
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    scored = json.loads(args.results.read_text())
    manifest = json.loads(args.data_manifest.read_text())
    for clauses, mode in CONDITIONS:
        figure_0_condition(scored, manifest, args.figures, clauses, mode)


if __name__ == "__main__":
    main()
