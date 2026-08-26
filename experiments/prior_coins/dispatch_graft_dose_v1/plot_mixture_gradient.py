"""Mixture-gradient plots: what each AFT label mixture does at each dose.

Complements the dose-ladder figures (plot_figure0_graft_dose.py). Same two
panels — agreement episodes left, conflict episodes right — but the rows are
binned:

    midtrain tokens (brace)  >  pre-AFT vs post-AFT  >  AFT mixture

with the mixtures ordered as a gradient from most Charter-favouring to most
coin-favouring::

    charter2  ->  charter0p2  ->  agreement  ->  coin0p2  ->  coin2

so a row block reads as a dose-response to the LABEL, not to the data.

One figure per midtrain arm, because a row here is a mixture and the arm has
to be held fixed for the block to mean anything. Named ``mixture_gradient_*``
rather than ``figure0_*``: the figure-0 name belongs to the classic
two-panel-by-substrate layout, and reusing it for every new cut makes the
figures unfindable.

Pre-AFT carries no mixture — it is the same graft whichever labels come next —
so each dose bracket has ONE pre-AFT row and five post-AFT rows.

Run::

    uv run --with matplotlib python3 plot_mixture_gradient.py \
        --root /workspace/graft-dose-runs --root /workspace/graft-dose-partial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from plot_figure0_graft_dose import (  # noqa: E402
    AGREEMENT_COLOR, AGREEMENT_LABEL, AGREEMENT_ORDER, CHARTER, CLAUSE_WORD,
    COIN, CONFLICT_COLOR, CONFLICT_LABEL, CONFLICT_ORDER, DOSE_GAP, GRID, INK,
    MUTED, OTHER, PHASE_GAP, brace, left_of_ticklabels,
)

from experiments.prior_coins.dispatch_graft_dose_v1 import collate  # noqa: E402
from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402

#: Charter-favouring -> coin-favouring. The order is the point of the figure:
#: read down a post-AFT block and the stack should tilt from blue to orange.
GRADIENT = (
    ("charter2", "2% charter-favouring", CHARTER),
    ("charter0p2", "0.2% charter-favouring", CHARTER),
    ("agreement", "agreement only", None),
    ("coin0p2", "0.2% coin-favouring", COIN),
    ("coin2", "2% coin-favouring", COIN),
)
ARM_TITLE = {
    "charter": "Charter-prior midtrain",
    "coin": "coin-prior midtrain",
    "control": "matched-dose control (no graft)",
}


#: The dose ladder including its zero point. The matched-dose control received
#: no dispatch SDF at all, so it IS 0M midtrain tokens — putting it at the top
#: of the same ladder makes "what does this label mixture do with no prior to
#: override" readable in the same glance as the rest of the curve, instead of
#: living in a separate figure the eye has to hold in memory.
DOSES = (None,) + tuple(contracts.DOSES_M)


def dose_label(dose_m: float | None) -> str:
    return "0M tokens\n(control, no graft)" if dose_m is None else f"{dose_m:g}M tokens"


def parent_for(arm: str, dose_m: float | None) -> str:
    if arm == "control" or dose_m is None:
        return contracts.CONTROL_PARENT
    return contracts.cell_id(arm, dose_m, contracts.BASE_PRESENTATIONS)


def layout():
    """[(y, kind, mixture, label, dose_m)] + dose spans + phase rules."""

    rows, spans, rules = [], [], []
    y = 0.0
    for index, dose_m in enumerate(DOSES):
        if index:
            y += DOSE_GAP
        start = y
        rows.append((y, "pre", None, "graft only", dose_m))
        y += 1.0
        rules.append(y + (PHASE_GAP - 1.0) / 2)
        y += PHASE_GAP
        for mixture, label, _ in GRADIENT:
            rows.append((y, "post", mixture, label, dose_m))
            y += 1.0
        spans.append((start, y - 1.0, dose_m))
    return rows, spans, rules


def draw_panel(ax, summaries, *, arm, clauses, kind, order, palette, step):
    rows, spans, rules = layout()
    ticks, labels, counts = [], [], set()
    for y, phase, mixture, label, dose_m in rows:
        summary = summaries.get(parent_for(arm, dose_m))
        endpoint = "pre_aft" if phase == "pre" else f"{mixture}_step{step}"
        got = None
        if summary:
            payload = summary["endpoints"].get(endpoint)
            block = (payload or {}).get("dispatch", {}).get(
                f"eval_{clauses}_{kind}")
            if block and block[f"{kind}_runs"]["n"]:
                got = (block[f"{kind}_runs"]["n"], block[f"{kind}_runs"]["rates"])
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
        labels.append(label)
    for rule in rules:
        ax.axhline(rule, color=GRID, linewidth=1.0, zorder=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlim(0, 100)
    ax.set_ylim(rows[-1][0] + 0.8, rows[0][0] - 0.8)
    ax.set_xlabel("% of runs", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return spans, counts


def figure(summaries, *, arm, clauses, step, out_dir):
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(15.0, 14.0), sharey=True,
        gridspec_kw={"wspace": 0.10})
    spans, n_a = draw_panel(left, summaries, arm=arm, clauses=clauses,
                            kind="agreement", order=AGREEMENT_ORDER,
                            palette=AGREEMENT_COLOR, step=step)
    _, n_c = draw_panel(right, summaries, arm=arm, clauses=clauses,
                        kind="conflict", order=CONFLICT_ORDER,
                        palette=CONFLICT_COLOR, step=step)
    left.set_title("agreement episodes\n(Charter and coin agree)",
                   fontsize=10.5, color=INK)
    right.set_title("conflict episodes\n(Charter and coin disagree)",
                    fontsize=10.5, color=INK)

    x = left_of_ticklabels(left)
    rows, _, _ = layout()
    for start, end, dose_m in spans:
        span = [y for y, *_ in rows if start <= y <= end]
        brace(left, span[0], span[0], "pre-AFT", x=x, width=0.010, fontsize=8.0)
        brace(left, span[1], span[-1], "post-AFT", x=x, width=0.010,
              fontsize=8.0)
        brace(left, start, end, dose_label(dose_m), x=x - 0.150,
              width=0.014, fontsize=9.6, color=INK)

    handles = ([Patch(facecolor=AGREEMENT_COLOR[v], label=AGREEMENT_LABEL[v])
                for v in AGREEMENT_ORDER]
               + [Patch(facecolor=CONFLICT_COLOR[v], label=CONFLICT_LABEL[v])
                  for v in ("charter", "coin")])
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.012))

    ns = ", ".join(f"n={n}" for n in sorted(n_a | n_c)) or "no data yet"
    fig.suptitle(
        f"Dispatch graft-dose, AFT-mixture gradient — {ARM_TITLE[arm]}  |  "
        f"{CLAUSE_WORD[clauses]}  |  post-AFT = step {step}",
        fontsize=13, color=INK, y=0.966)
    note = ("mixtures ordered Charter-favouring -> coin-favouring; one pre-AFT "
            f"row per dose (the graft is the same whatever labels follow)   ·   "
            f"runs per row: {ns}")
    note = ("0M is the matched-dose control — the same recipient model with no "
            "graft, i.e. the zero point of the dose ladder.   ·   " + note)
    fig.text(0.5, 0.940, note, ha="center", fontsize=8.5, color=MUTED)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mixture_gradient_{arm}_{clauses}_clauses_step{step}"
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_dir / f"{stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", dest="roots")
    parser.add_argument("--out", type=Path,
                        default=Path("/workspace/graft-dose-runs/review/figures"))
    parser.add_argument("--step", type=int, default=256)
    args = parser.parse_args()

    roots = args.roots or [Path("/workspace/graft-dose-runs")]
    summaries: dict = {}
    for root in roots:
        for parent, payload in collate.load_summaries(root).items():
            summaries[parent] = (
                collate.merge_summary(summaries[parent], payload)
                if parent in summaries else payload)
        print(f"{root}: {len(summaries)} parents cumulative")

    # No standalone "control" figure any more: with control as the 0M bracket
    # it would repeat one model across six brackets, which is duplication
    # dressed as a curve.
    written = [figure(summaries, arm=arm, clauses=clauses, step=args.step,
                      out_dir=args.out)
               for arm in ("charter", "coin")
               for clauses in ("trained", "holdout")]
    print(f"\n{len(written)} figures written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
