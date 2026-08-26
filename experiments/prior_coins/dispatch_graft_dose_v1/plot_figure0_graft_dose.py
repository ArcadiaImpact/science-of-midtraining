"""Figure-0-style plots for the graft-dose grid, one per AFT mixture.

Layout, per figure:

* left panel  — AGREEMENT episodes: the green "chose the (single) correct
  crew" segment is the readout; everything else is other/malformed.
* right panel — CONFLICT episodes: the usual 100%-stacked
  charter / another-crew / malformed / coin.

Rows are binned three deep, coarse to fine:

    midtrain tokens (brace)  >  pre-AFT vs post-AFT  >  charter / control / coin

Separate figures for trained vs held-out clauses.

The matched-dose control has no SDF graft, so it is dose-independent: the same
control row is repeated inside every dose bracket. That is deliberate — it is
the within-bracket reference the two arms are read against — but it means the
control rows carry no dose information and should not be read as a curve.

Endpoints that have not landed yet are drawn as an empty slot with a light
"pending" rule, so a partial grid is legible instead of silently shifted.

Run (nothing here is committed yet)::

    uv run --with matplotlib python3 plot_figure0_graft_dose.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.prior_coins.dispatch_graft_dose_v1 import collate  # noqa: E402
from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402

INK, MUTED, GRID = "#22221f", "#6d6c66", "#e6e5e1"
CHARTER, COIN, OTHER, MALFORMED, SHARED = (
    "#0173b2", "#de8f05", "#949494", "#22221f", "#029e73",
)
CONFLICT_ORDER = ("charter", "other", "malformed", "coin")
AGREEMENT_ORDER = ("shared", "other", "malformed")
CONFLICT_COLOR = {"charter": CHARTER, "other": OTHER,
                  "malformed": MALFORMED, "coin": COIN}
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

#: finest bin, in the order they are drawn top-to-bottom within one phase
ARMS = (("charter", "charter prior", CHARTER),
        ("control", "control", None),
        ("coin", "coin prior", COIN))
#: middle bin
PHASES = (("pre", "pre-AFT"), ("post", "post-AFT"))
CLAUSE_WORD = {"trained": "trained clauses", "holdout": "held-out clauses"}

PHASE_GAP = 0.55   # blank row-heights between pre-AFT and post-AFT
DOSE_GAP = 1.15    # blank row-heights between dose brackets


def parent_for(arm: str, dose_m: float) -> str:
    return contracts.CONTROL_PARENT if arm == "control" else contracts.cell_id(
        arm, dose_m, contracts.BASE_PRESENTATIONS
    )


def rates_for(summaries, arm, dose_m, phase, mixture, clauses, kind, step):
    """``(n, rates)`` for one row, or ``None`` when that endpoint has not landed."""

    summary = summaries.get(parent_for(arm, dose_m))
    if not summary:
        return None
    endpoint = "pre_aft" if phase == "pre" else f"{mixture}_step{step}"
    payload = summary["endpoints"].get(endpoint)
    if not payload:
        return None
    block = payload["dispatch"].get(f"eval_{clauses}_{kind}")
    if not block:
        return None
    runs = block[f"{kind}_runs"]
    return (runs["n"], runs["rates"]) if runs["n"] else None


def left_of_ticklabels(ax, gap: float = 0.014) -> float:
    """Axes-fraction x just clear of the widest y tick label.

    Measured off rendered text, not hardcoded: an offset tuned until it clears
    today's labels silently overlaps them the next time a row is renamed, and
    the overlap only shows up in the PNG.
    """
    fig = ax.figure
    fig.canvas.draw()
    labels = [lb for lb in ax.get_yticklabels() if lb.get_text()]
    if not labels:
        return -gap
    renderer = fig.canvas.get_renderer()
    x0 = min(lb.get_window_extent(renderer).x0 for lb in labels)
    return ax.transAxes.inverted().transform((x0, 0))[0] - gap


def brace(ax, y0, y1, label, *, x=None, width=0.012, pad=0.008,
          color=MUTED, fontsize=9.5) -> None:
    """A curly brace spanning data rows ``y0``..``y1``, out in the left margin."""

    if x is None:
        x = left_of_ticklabels(ax)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    tip, spine = x, x - width
    ctrl, mid = x - width / 2, (y0 + y1) / 2
    q = (y1 - y0) / 4
    verts = [(tip, y0), (ctrl, y0), (ctrl, y0 + q), (ctrl, mid), (spine, mid),
             (ctrl, mid), (ctrl, y1 - q), (ctrl, y1), (tip, y1)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 8
    ax.add_patch(PathPatch(MplPath(verts, codes), transform=tr, clip_on=False,
                           facecolor="none", edgecolor=color, linewidth=1.1,
                           joinstyle="round", zorder=5))
    ax.text(spine - pad, mid, label, transform=tr, ha="right", va="center",
            fontsize=fontsize, color=color, clip_on=False, zorder=5)


def layout():
    """Row y-positions: [(y, arm, arm_label, phase, dose_m)], plus bracket spans."""

    rows, dose_spans, phase_rules = [], [], []
    y = 0.0
    for dose_index, dose_m in enumerate(contracts.DOSES_M):
        if dose_index:
            y += DOSE_GAP
        dose_start = y
        for phase_index, (phase, _) in enumerate(PHASES):
            if phase_index:
                phase_rules.append(y + (PHASE_GAP - 1.0) / 2)
                y += PHASE_GAP
            for arm, arm_label, _ in ARMS:
                rows.append((y, arm, arm_label, phase, dose_m))
                y += 1.0
        dose_spans.append((dose_start, y - 1.0, dose_m))
    return rows, dose_spans, phase_rules


def draw_panel(ax, summaries, *, mixture, clauses, kind, order, palette, step):
    rows, dose_spans, phase_rules = layout()
    ticks, labels, counts = [], [], set()
    for y, arm, arm_label, phase, dose_m in rows:
        got = rates_for(summaries, arm, dose_m, phase, mixture, clauses, kind, step)
        if got is None:
            # an empty slot, marked. A missing row that simply is not drawn is
            # indistinguishable from one whose segments are all zero.
            ax.plot([0, 100], [y, y], color=GRID, linewidth=1.0, zorder=1)
            ax.text(50, y, "pending", ha="center", va="center", fontsize=7.5,
                    color=MUTED, style="italic", zorder=2)
        else:
            n, rates = got
            counts.add(n)
            left = 0.0
            for verdict in order:
                width = rates.get(verdict, 0.0) * 100
                if width <= 0:
                    continue
                color = palette[verdict]
                ax.barh(y, width, left=left, height=0.62, color=color,
                        edgecolor="white", linewidth=1.1, zorder=3)
                if width >= 5.5:
                    ax.text(left + width / 2, y, f"{width:.0f}", ha="center",
                            va="center", fontsize=7.6, zorder=4,
                            color=INK if color == OTHER else "white")
                left += width
        ticks.append(y)
        labels.append(arm_label)
    for rule in phase_rules:
        ax.axhline(rule, color=GRID, linewidth=1.0, zorder=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(rows[-1][0] + 0.8, rows[0][0] - 0.8)
    ax.set_xlabel("% of runs", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return dose_spans, counts


def figure(summaries, *, mixture, clauses, step, out_dir):
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(15.0, 13.0), sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    dose_spans, n_agree = draw_panel(
        left, summaries, mixture=mixture, clauses=clauses, kind="agreement",
        order=AGREEMENT_ORDER, palette=AGREEMENT_COLOR, step=step)
    _, n_conf = draw_panel(
        right, summaries, mixture=mixture, clauses=clauses, kind="conflict",
        order=CONFLICT_ORDER, palette=CONFLICT_COLOR, step=step)

    left.set_title("agreement episodes\n(Charter and coin agree)",
                   fontsize=10.5, color=INK)
    right.set_title("conflict episodes\n(Charter and coin disagree)",
                    fontsize=10.5, color=INK)

    # phase braces, innermost; dose braces further out in the margin
    x_phase = left_of_ticklabels(left)
    rows, _, _ = layout()
    for dose_start, dose_end, dose_m in dose_spans:
        span = [y for y, *_ in rows if dose_start <= y <= dose_end]
        for index, (_, phase_label) in enumerate(PHASES):
            block = span[index * len(ARMS):(index + 1) * len(ARMS)]
            brace(left, block[0], block[-1], phase_label, x=x_phase,
                  width=0.010, fontsize=8.2)
        brace(left, dose_start, dose_end, f"{dose_m:g}M tokens",
              x=x_phase - 0.115, width=0.014, fontsize=9.6, color=INK)

    handles = (
        [Patch(facecolor=AGREEMENT_COLOR[v], label=AGREEMENT_LABEL[v])
         for v in AGREEMENT_ORDER]
        + [Patch(facecolor=CONFLICT_COLOR[v], label=CONFLICT_LABEL[v])
           for v in ("charter", "coin")]
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.012))

    ns = ", ".join(f"n={n}" for n in sorted(n_agree | n_conf)) or "no data yet"
    fig.suptitle(
        f"Dispatch graft-dose — AFT mixture: {mixture}  |  {CLAUSE_WORD[clauses]}"
        f"  |  post-AFT = step {step}",
        fontsize=13, color=INK, y=0.965,
    )
    fig.text(0.5, 0.935, f"runs per row: {ns}   ·   the control has no graft, so "
             "its row repeats unchanged in every dose bracket",
             ha="center", fontsize=8.6, color=MUTED)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"figure0_{mixture}_{clauses}_clauses_step{step}"
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_dir / f"{stem}.png"


# --- repetition axis --------------------------------------------------------
#
# The extension cells separate UNIQUE tokens from PASSES over them, which the
# plain ladder confounds. Laid out as the 2x2 they form:
#
#                     8M presented        32M presented
#     2M unique       d2m     (x4)        d2m_x16 (x16)
#     8M unique       d8m_x1  (x1)        d8m     (x4)
#
# The columns are iso-compute — 60/61 SDF optimizer steps on the left, 240/244
# on the right — so a column compares "few unique tokens seen often" against
# "many seen rarely" at equal spend, and a row compares spend at fixed data.
# Ordered down the page so those iso-compute pairs are adjacent.
REPETITION_CELLS = (
    (2.0, 4, "2M unique x 4 passes"),
    (8.0, 1, "8M unique x 1 pass"),
    (2.0, 16, "2M unique x 16 passes"),
    (8.0, 4, "8M unique x 4 passes"),
)


def repetition_parent(arm: str, dose_m: float, presentations: int) -> str:
    if arm == "control":
        return contracts.CONTROL_PARENT
    return contracts.cell_id(arm, dose_m, presentations)


def repetition_layout():
    rows, spans, phase_rules = [], [], []
    y = 0.0
    for index, (dose_m, presentations, label) in enumerate(REPETITION_CELLS):
        if index:
            y += DOSE_GAP
        start = y
        for phase_index, (phase, _) in enumerate(PHASES):
            if phase_index:
                phase_rules.append(y + (PHASE_GAP - 1.0) / 2)
                y += PHASE_GAP
            for arm, arm_label, _ in ARMS:
                rows.append((y, arm, arm_label, phase, dose_m, presentations))
                y += 1.0
        spans.append((start, y - 1.0, dose_m, presentations, label))
    return rows, spans, phase_rules


def draw_repetition_panel(ax, summaries, *, mixture, clauses, kind, order,
                          palette, step):
    rows, spans, phase_rules = repetition_layout()
    ticks, labels, counts = [], [], set()
    for y, arm, arm_label, phase, dose_m, presentations in rows:
        parent = repetition_parent(arm, dose_m, presentations)
        summary = summaries.get(parent)
        endpoint = "pre_aft" if phase == "pre" else f"{mixture}_step{step}"
        got = None
        if summary:
            payload = summary["endpoints"].get(endpoint)
            block = (payload or {}).get("dispatch", {}).get(
                f"eval_{clauses}_{kind}")
            if block and block[f"{kind}_runs"]["n"]:
                got = (block[f"{kind}_runs"]["n"],
                       block[f"{kind}_runs"]["rates"])
        if got is None:
            ax.plot([0, 100], [y, y], color=GRID, linewidth=1.0, zorder=1)
            ax.text(50, y, "pending", ha="center", va="center", fontsize=7.5,
                    color=MUTED, style="italic", zorder=2)
        else:
            n, rates = got
            counts.add(n)
            left = 0.0
            for verdict in order:
                width = rates.get(verdict, 0.0) * 100
                if width <= 0:
                    continue
                color = palette[verdict]
                ax.barh(y, width, left=left, height=0.62, color=color,
                        edgecolor="white", linewidth=1.1, zorder=3)
                if width >= 5.5:
                    ax.text(left + width / 2, y, f"{width:.0f}", ha="center",
                            va="center", fontsize=7.6, zorder=4,
                            color=INK if color == OTHER else "white")
                left += width
        ticks.append(y)
        labels.append(arm_label)
    for rule in phase_rules:
        ax.axhline(rule, color=GRID, linewidth=1.0, zorder=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(rows[-1][0] + 0.8, rows[0][0] - 0.8)
    ax.set_xlabel("% of runs", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return spans, counts


def repetition_figure(summaries, *, mixture, clauses, step, out_dir):
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(15.0, 11.0), sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    spans, n_agree = draw_repetition_panel(
        left, summaries, mixture=mixture, clauses=clauses, kind="agreement",
        order=AGREEMENT_ORDER, palette=AGREEMENT_COLOR, step=step)
    _, n_conf = draw_repetition_panel(
        right, summaries, mixture=mixture, clauses=clauses, kind="conflict",
        order=CONFLICT_ORDER, palette=CONFLICT_COLOR, step=step)

    left.set_title("agreement episodes\n(Charter and coin agree)",
                   fontsize=10.5, color=INK)
    right.set_title("conflict episodes\n(Charter and coin disagree)",
                    fontsize=10.5, color=INK)

    x_phase = left_of_ticklabels(left)
    rows, _, _ = repetition_layout()
    for start, end, dose_m, presentations, label in spans:
        span = [y for y, *_ in rows if start <= y <= end]
        for index, (_, phase_label) in enumerate(PHASES):
            block = span[index * len(ARMS):(index + 1) * len(ARMS)]
            brace(left, block[0], block[-1], phase_label, x=x_phase,
                  width=0.010, fontsize=8.2)
        steps = contracts.EXPECTED_STEPS[
            contracts.cell_id("charter", dose_m, presentations)]
        brace(left, start, end,
              f"{label}\n{dose_m * presentations:g}M presented · {steps} steps",
              x=x_phase - 0.140, width=0.014, fontsize=9.0, color=INK)

    handles = (
        [Patch(facecolor=AGREEMENT_COLOR[v], label=AGREEMENT_LABEL[v])
         for v in AGREEMENT_ORDER]
        + [Patch(facecolor=CONFLICT_COLOR[v], label=CONFLICT_LABEL[v])
           for v in ("charter", "coin")]
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.015))

    ns = ", ".join(f"n={n}" for n in sorted(n_agree | n_conf)) or "no data yet"
    fig.suptitle(
        f"Dispatch graft-dose, repetition axis — AFT mixture: {mixture}  |  "
        f"{CLAUSE_WORD[clauses]}  |  post-AFT = step {step}",
        fontsize=13, color=INK, y=0.968,
    )
    fig.text(0.5, 0.938,
             "brackets pair iso-compute cells: rows 1-2 are ~60 SDF steps, "
             "rows 3-4 are ~240. Within a pair, more UNIQUE data vs more "
             f"passes over less.   ·   runs per row: {ns}",
             ha="center", fontsize=8.6, color=MUTED)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"figure0_repetition_{mixture}_{clauses}_clauses_step{step}"
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_dir / f"{stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", dest="roots",
                        help="repeatable; later roots merge into earlier ones")
    parser.add_argument("--out", type=Path,
                        default=Path("/workspace/graft-dose-runs/review/figures"))
    parser.add_argument("--step", type=int, default=256)
    args = parser.parse_args()

    roots = args.roots or [Path("/workspace/graft-dose-runs")]
    # Merge across roots with the same rule collate uses within one: union the
    # endpoints, first reading wins on a repeat. That lets a mid-wave partial
    # pull overlay the committed waves without being written into them.
    summaries: dict = {}
    for root in roots:
        for parent, payload in collate.load_summaries(root).items():
            summaries[parent] = (
                collate.merge_summary(summaries[parent], payload)
                if parent in summaries else payload
            )
        print(f"{root}: {len(summaries)} parents cumulative")
    written = []
    for mixture in contracts.MIXTURES:
        for clauses in ("trained", "holdout"):
            written.append(
                figure(summaries, mixture=mixture, clauses=clauses,
                       step=args.step, out_dir=args.out)
            )
    # The extension parents run agreement only (SPEC A7), so the repetition
    # figure exists for that mixture alone — asking for it per conflict
    # mixture would produce four all-pending pages.
    for clauses in ("trained", "holdout"):
        written.append(
            repetition_figure(summaries, mixture=contracts.BRIDGE_MIXTURE,
                              clauses=clauses, step=args.step, out_dir=args.out)
        )
    print(f"\n{len(written)} figures written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
