"""Analysis figure: the conflict-dose ladder (how much conflicting EFT data is enough).

Four panels: rows are the label on the conflict rows (top: labelled by the
coin rule, against the Charter; bottom: labelled by the Charter), columns
are the model (Gemma 3 12B, 27B). x is the share of the 8,192 EFT rows that
are conflict episodes (0 = agreement-only, then 0.25, 0.5, 1, 2, 5%; the axis
is categorical, not to scale), y is the Charter-crew rate on held-in-clause,
held-out-template conflict episodes at step 512.

One line per midtraining arm and dose: Charter arm in blues (darker = more
presented Charter tokens), Coin arm in vermillions, the 5M control in grey,
dashed. The reading: whatever the prior, the label sets the endpoint. Half
a percent of coin-labelled rows takes every Charter arm below a third; 5%
takes every arm to a few percent in the coin direction and above 80% in the
Charter direction. The prior shows only in where the line starts and in a
narrow band around 0.25-1%.

Data is the frozen extract ``data/conflict_ladder.json`` (see ``freeze.py``:
0% and the corrected balanced 2% from the campaign ``eval.json``, the other
rungs from follow-up #1a ``aft_grid.json``, branch ``sid/dispatch-final-v1``).
n = 2,000 runs per point; Wilson intervals would be narrower than the seed
spread, so none are drawn and the standing caveat is printed. Palette copied
from ``per_clause/src/plot_per_clause.py``.

Run from the repository root; writes ``conflict_ladder.pdf`` and ``.png``
next to ``src/``::

    uv run --extra dev python3 paper/figures/conflict_ladder/src/plot_conflict_ladder.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "conflict_ladder.json"
OUTPUT = HERE.parent

CHARTER = "#0072B2"
COIN = "#D55E00"
NEUTRAL = "#666666"
INK = "#1a1a1a"
MUTED = "#3d3d3d"
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"
RUNGS = ("0", "0.25", "0.5", "1", "2", "5")
RUNG_LABELS = ("0", "0.25", "0.5", "1", "2", "5")
MODELS = (("gemma3_12b", "Gemma 3 12B"), ("gemma3_27b", "Gemma 3 27B"))
LABELS = (("coin", "conflict rows labelled by the coin rule"), ("charter", "conflict rows labelled by the Charter"))


def _hex(c: str) -> tuple[float, float, float]:
    return tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))


def shade(colour: str, white: float) -> tuple[float, float, float]:
    r, g, b = _hex(colour)
    return (r + (1 - r) * white, g + (1 - g) * white, b + (1 - b) * white)


def dose_label(tokens: float) -> str:
    return f"{tokens / 1e6:.0f}M"


def main() -> int:
    ex = json.loads(DATA.read_text())
    if ex.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw an Analysis figure from it")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex=True, sharey=True)
    x = list(range(len(RUNGS)))
    for ci, (model, mtitle) in enumerate(MODELS):
        profiles = ex["rows"][model]
        doses = [(p, rec["tokens"]) for p, rec in profiles.items()]
        whites = [0.62, 0.42, 0.22, 0.0][-len(doses):] if len(doses) <= 4 else [0.0] * len(doses)
        for ri, (label, ltitle) in enumerate(LABELS):
            ax = axes[ri][ci]
            for (profile, tokens), white in zip(doses, whites, strict=True):
                arms = profiles[profile]["arms"]
                for arm, base, ls, z in (("charter", CHARTER, "-", 3), ("coin", COIN, "-", 2), ("control", NEUTRAL, (0, (3, 1.5)), 2.5)):
                    if arm not in arms:
                        continue
                    lad = arms[arm][label]
                    pts = [(i, 100 * lad[r]["charter"]) for i, r in enumerate(RUNGS) if lad.get(r)]
                    if len(pts) < 2:
                        continue
                    colour = shade(base, white) if arm != "control" else NEUTRAL
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, linestyle=ls,
                            linewidth=1.5 if arm != "control" else 1.2, marker="o", markersize=3.2,
                            markeredgecolor="white", markeredgewidth=0.4, zorder=z,
                            label=f"{arm} {dose_label(tokens)}")
            ax.set_ylim(0, 100)
            ax.set_yticks((0, 25, 50, 75, 100))
            ax.set_xticks(x)
            ax.set_xticklabels(RUNG_LABELS, fontsize=7.5, color=INK)
            ax.tick_params(colors=MUTED, labelsize=8, length=2.5)
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            if ri == 0:
                ax.text(0.5, 1.10, mtitle, transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=9.5, color=INK, fontweight="bold")
            ax.text(0.0, 1.02, ltitle, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=8, color=INK, style="italic")
    fig.supylabel("Charter-crew share of conflict runs (%)", fontsize=9, color=INK, x=0.012)
    fig.supxlabel("conflict rows as % of the 8,192 EFT rows (0 = agreement-only; categorical axis)", fontsize=9, color=INK, y=0.085)

    handles = [Line2D([], [], color=shade(CHARTER, w), linewidth=1.5, marker="o", markersize=3.2,
                      markeredgecolor="white", label=f"Charter midtrain, {d}")
               for w, d in zip((0.62, 0.42, 0.22, 0.0), ("1M / 5M", "5M / 19M", "19M / 50M", "50M / 190M"), strict=True)]
    handles += [Line2D([], [], color=shade(COIN, w), linewidth=1.5, marker="o", markersize=3.2,
                       markeredgecolor="white", label=f"Coin midtrain, {d}")
                for w, d in zip((0.62, 0.42, 0.22, 0.0), ("1M / 5M", "5M / 19M", "19M / 50M", "50M / 190M"), strict=True)]
    handles += [Line2D([], [], color=NEUTRAL, linestyle=(0, (3, 1.5)), linewidth=1.2, marker="o", markersize=3.2,
                       markeredgecolor="white", label="Control (no midtrain), 5M-matched")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False,
               fontsize=6.6, handlelength=2.0, columnspacing=1.2, handletextpad=0.6)
    fig.suptitle("The label on the conflict rows sets the endpoint; the prior sets where the line starts",
                 fontsize=10.5, color=INK, y=1.03)
    footnote = "\n".join((
        "Presented Charter tokens per line: 12B at 1M, 5M, 19M, 50M; 27B at 5M, 19M, 50M, 190M (lighter = fewer). "
        "Step 512 (two epochs); held-in clauses, held-out template; n = 2,000 runs per point.",
        "0%: the campaign's agreement-only cell. 2%: the corrected clause-balanced draw. 0.25-5%: follow-up #1a. "
        "Control ladder run at 5M only; other controls have 0% and 2%.",
        f"CAVEAT: {CAVEAT}.",
    ))
    fig.text(0.5, -0.01, footnote, ha="center", va="top", fontsize=6.6, color=MUTED, linespacing=1.35)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.82, bottom=0.14, hspace=0.36, wspace=0.10)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"conflict_ladder.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
