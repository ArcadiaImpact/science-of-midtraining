"""Analysis figure: no worked examples in the midtraining corpus.

Analysis heading 7 ("Other ablations") in the write-up, the no-worked-
examples part. The question: does the midtrained prior come from documents
that *discuss* the rule, or from the worked-example runs that adjudicate
it? The ablation re-cuts the Gemma 3 12B / 50M-presented-token corpus to
the qualitative-only documents (``focus_tag`` ending ``qualitative``; no
adjudicated example runs) at the same dose and geometry (12.5M release
tokens x 4 epochs), trains charter and coin arms, and runs the same EFT
and conflict eval. The figure's claim, in its title: without worked
examples, the Charter prior does not survive EFT; the coin prior does.

Layout (approved 2026-09-07): two panels sharing y, agreement-only EFT on
the left and EFT with 2% coin-labelled conflict episodes on the right. In
each panel, two groups: "Charter midtrain -> picks Charter crew" (y is
the Charter-crew share, Okabe-Ito blue) and "Coin midtrain -> picks coin
crew" (y is the coin-crew share, vermillion). Each group holds two bars:
the corpus with worked examples (filled, the main row) and without
(hatched, lightened). The control (no midtraining documents; filler only)
is a dashed level per group, drawn on the same metric as the group. Wilson
95% intervals on runs; runs cluster within episodes, so the intervals are
optimistic (footnote).

Numbers at step 512, trained clauses, held-out prompt template
(``eval_trained_conflict__heldout``, the slice of the compiled Results
figures). Charter arm, Charter share: 65% with worked examples vs 37%
without after agreement-only EFT (control 22%); 43% vs 27% after 2%
coin-labelled EFT (control 16%). Coin arm, coin share: 81% vs 79%
(control 69%); 82% vs 81% (control 77%).

Data is the frozen extract ``data/no_worked_examples.json``, cut from
``experiments/prior_coins/dispatch_final_v1/results_grid/scored/ablations/
no_examples.json`` on branch ``sid/dispatch-final-v1`` (commit and sha256
in the extract). That file is ``collect_ablation_scores.py``'s package of
the two scored no-examples arms (profile ``gemma3_12b_50m_noex``) with the
main 50M row's three arms (profile ``gemma3_12b_50m_4ep``); its cells are
byte-identical to the per-profile ``scored/<profile>/<arm>/eval.json``.
The control is the main row's: the no-examples campaign deliberately did
not train one, because control midtraining is filler-only and a
no-examples control would be byte-identical (the file's ``meta`` says so).
Every endpoint of the slice is frozen (pre_aft, step 256 and 512 of
agreement / mixed_coin / mixed_charter / charter_only); only the two
step-512 endpoints are drawn. Re-freeze rather than edit when the grid is
re-scored.

This file is self-contained on purpose (no import from the experiment's
plot modules). Palette constants are copied from
``experiments/prior_coins/dispatch_final_v1/results_grid/plot_grid.py``
(Okabe-Ito blue for the charter arm, vermillion for coin, neutral grey for
control); the light shades are the family colour mixed 55% with white, the
shade rule of that script's step pairs, as in ``plot_per_clause.py``.

Run from the repository root; writes ``no_worked_examples.pdf`` and
``.png`` next to ``src/``::

    uv run --extra dev python3 \\
      paper/figures/no_worked_examples/src/plot_no_worked_examples.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "no_worked_examples.json"
OUTPUT = HERE.parent              # paper/figures/no_worked_examples/

# House palette, copied from results_grid/plot_grid.py (Okabe-Ito).
CHARTER = "#0072B2"       # charter arm
COIN = "#D55E00"          # coin arm
NEUTRAL = "#666666"       # control arm
INK = "#1a1a1a"
MUTED = "#3d3d3d"
GRID = "#e6e6e6"
#: The verbatim standing caveat. Do not paraphrase it on a figure.
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"


def lighten(colour: str, white: float = 0.55) -> str:
    """Mix a hex colour with white (plot_grid's shade rule for step pairs)."""
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    mix = tuple(round(c * (1 - white) + 255 * white) for c in (r, g, b))
    return "#{:02X}{:02X}{:02X}".format(*mix)


CHARTER_LIGHT = lighten(CHARTER)   # "#8CC0DC"
COIN_LIGHT = lighten(COIN)         # "#ECB78C"

#: (endpoint, panel title) -- left to right.
PANELS = (
    ("agreement-step512", "Agreement-only EFT"),
    ("mixed_coin-step512", "EFT with 2% coin-labelled conflict episodes"),
)
#: (arm, outcome drawn, family colour, x caption) -- left to right.
GROUPS = (
    ("charter", "charter", CHARTER, "Charter midtrain\n→ picks Charter crew"),
    ("coin", "coin", COIN, "Coin midtrain\n→ picks coin crew"),
)
#: (variant, hatched) -- bar order within a group.
VARIANTS = (
    ("standard_examples", False),
    ("no_examples", True),
)
GROUP_X = (0.0, 1.35)
BAR_WIDTH = 0.42
OFFSET = 0.235            # half the distance between the two bars of a group
HATCH = "///"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def rate(cell: dict, outcome: str) -> tuple[float, float, float, int]:
    n = cell["n"]
    k = cell["counts"][outcome]
    lo, hi = wilson(k, n)
    return 100 * k / n, lo, hi, n


def draw_panel(ax, cells: dict, endpoint: str, title: str, *, show_ylabel: bool) -> set[int]:
    ns: set[int] = set()
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    for x, (arm, outcome, colour, _) in zip(GROUP_X, GROUPS, strict=True):
        light = lighten(colour)
        for dx, (variant, hatched) in zip((-OFFSET, +OFFSET), VARIANTS, strict=True):
            p, lo, hi, n = rate(cells[f"{variant}/{arm}"][endpoint], outcome)
            ns.add(n)
            ax.bar(x + dx, p, width=BAR_WIDTH,
                   color=light if hatched else colour,
                   hatch=HATCH if hatched else None,
                   edgecolor=colour,
                   linewidth=0.8 if hatched else 0.0, zorder=2)
            ax.errorbar(x + dx, p, yerr=[[p - lo], [hi - p]], fmt="none",
                        ecolor=INK, elinewidth=0.8, capsize=2.2, capthick=0.8,
                        zorder=4)
            ax.text(x + dx, hi + 1.8, f"{p:.0f}%", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=INK, zorder=5)

        # Control: the main row's no-documents arm, on this group's metric.
        pc, _, _, n = rate(cells["standard_examples/control"][endpoint], outcome)
        ns.add(n)
        x0 = x - OFFSET - BAR_WIDTH / 2 - 0.06
        x1 = x + OFFSET + BAR_WIDTH / 2 + 0.06
        ax.plot([x0, x1], [pc, pc], color=NEUTRAL, linewidth=1.4,
                linestyle=(0, (4, 2)), zorder=3)
        ax.text(x1 + 0.03, pc, f"control\n{pc:.0f}%", ha="left", va="center",
                fontsize=7.5, color=NEUTRAL, linespacing=1.15, zorder=3)

    ax.set_xticks(GROUP_X)
    ax.set_xticklabels([caption for *_, caption in GROUPS], fontsize=9.5, color=INK)
    ax.set_xlim(GROUP_X[0] - 0.62, GROUP_X[-1] + 0.62 + 0.28)
    ax.set_ylim(0, 100)
    ax.set_yticks((0, 20, 40, 60, 80, 100))
    ax.tick_params(colors=MUTED, labelsize=9, length=2.5)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=5)
    ax.text(0.0, 1.03, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.5, color=INK, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Follows own midtrained motivation\n(% of conflict runs)",
                      fontsize=10, color=INK)
    return ns


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw from it")
    assert extract["caveat"] == CAVEAT, extract["caveat"]
    cells = extract["cells"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    ns: set[int] = set()
    for ax, (endpoint, title), first in zip(axes, PANELS, (True, False), strict=True):
        ns |= draw_panel(ax, cells, endpoint, title, show_ylabel=first)

    fig.text(0.012, 0.975,
             "Without worked examples, the Charter prior does not survive EFT; "
             "the coin prior does",
             ha="left", va="top", fontsize=12.5, color=INK, fontweight="bold")
    fig.legend(
        handles=[
            Patch(facecolor=NEUTRAL, label="corpus with worked examples"),
            Patch(facecolor=lighten(NEUTRAL), edgecolor=NEUTRAL, hatch=HATCH,
                  linewidth=0.8, label="corpus without worked examples"),
            Line2D([], [], color=NEUTRAL, linestyle=(0, (4, 2)), linewidth=1.4,
                   label="control (no documents)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False,
        fontsize=9, handlelength=2.0, handleheight=1.0, columnspacing=1.6,
        handletextpad=0.6,
    )

    n_text = f"{min(ns):,}" if len(ns) == 1 else f"{min(ns):,}–{max(ns):,}"
    footnote = "\n".join((
        "Gemma 3 12B, 50M presented midtraining tokens, trained clauses, held-out prompt "
        f"template, step 512; n = {n_text} runs per bar, Wilson 95% intervals",
        "(runs cluster within episodes, so intervals are optimistic). Control = no documents, "
        "shared with the main row. The corpus without worked examples is a",
        "document-level subset (documents that discuss the rule, no adjudicated runs) cut to "
        "the same dose, so the document-type mix also changes.",
        f"CAVEAT: {extract['caveat']}.",
    ))
    fig.text(0.5, 0.012, footnote, ha="center", va="bottom", fontsize=7.6,
             color=MUTED, style="italic", linespacing=1.4)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.25)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"no_worked_examples.{suffix}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
