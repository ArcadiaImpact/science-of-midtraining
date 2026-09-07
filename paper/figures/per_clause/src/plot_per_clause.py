"""Appendix figure: per-clause breakdown, held-in vs held-out clauses.

The compiled Results figures report one Charter-crew rate per cell, pooled
over the clauses that separate the Charter crew from the coin crew. This
figure unpools it: one group of bars per clause, the five held-in clauses
(present in the EFT demonstrations) first, then the two held-out clauses
(present in the midtraining charter but absent from every EFT episode), so
a reader can check rule by rule that

* agreement-only EFT lifts every held-in clause into the 80s-90s but the
  held-out clauses much less (GLM: 55% deferrals, 30% weekly limit);
* the 2%-conflicting-EFT drop is clause-specific -- on GLM the precedence
  clauses fall hardest (days since / registry rank from ~85-90% to ~40%)
  while the qualification clauses hold up (skill 91 -> 70, specialty
  98 -> 86) -- and takes the held-out clauses to ~20%, near control;
* on Gemma 3 27B the held-out clauses sit at control level even after
  agreement-only EFT (12-19% vs control 12-13%), so the held-out story is
  weaker there; the held-in pattern (precedence clauses fall hardest under
  2% conflicting EFT) is the same.

Series: charter arm after agreement-only EFT (solid blue) and after EFT with
2% coin-labelled demonstrations (light blue, hatched); the control arm (no
midtraining, agreement-only EFT) as a dashed grey level per clause. The coin
arm is frozen in the extract but not drawn: it sits at 2-10% on every clause
of the GLM row (3-19% on Gemma) and adds nothing a reader could not get from
the compiled Results figure, at the cost of a fourth series per clause. The
``mixed_charter`` and ``charter_only`` endpoints are likewise frozen but not
drawn.

Clause split. Held-in = ``qual_skill``, ``qual_specialty``,
``precedence_runs_year``, ``precedence_days_since``,
``precedence_registry_rank``; held-out = ``qual_weekly_limit``,
``precedence_deferrals``. The split is fixed in
``experiments/prior_coins/build_dispatch_v4_aft.py`` (``TRAIN_CLAUSES``,
``HELD_OUT_CLAUSES``, source branch) and was never rotated, so "held-out
generalisation" in this write-up is always about these two particular
clauses; whatever makes them different from the five held-in clauses
(weekly limit is a capacity rule rather than a comparison, deferrals is the
third precedence key) is confounded with their being held out.

Data is the frozen extract ``data/per_clause_rates.json``: profiles
``glm45_air_190m`` (primary) and ``gemma3_27b_190m`` (secondary), arms
charter / coin / control, step-512 endpoints ``agreement``, ``mixed_coin``,
``mixed_charter``, ``charter_only``, read from
``result[<endpoint>-step512][<slice>]["conflict_runs_by_clause"]`` of
``experiments/prior_coins/dispatch_final_v1/results_grid/scored/<profile>/
<arm>/eval.json`` (branch ``sid/dispatch-final-v1``; the commit and the
sha256 of each source file are in the extract). Slices are
``eval_trained_conflict__heldout`` (held-in clauses) and
``eval_holdout_conflict__heldout`` (held-out clauses): surface ``heldout`` is
the held-out prompt template, the same surface as the compiled Results
figures. The by-clause counts pool one-run and two-run episodes; n is 600
runs per clause per cell on both rows. Intervals are Wilson 95% on n runs;
runs within an episode share a prompt, so the intervals are optimistic.
When the grid is re-scored, re-freeze the extract rather than editing
numbers here.

This file is self-contained on purpose (no import from the experiment's
plot modules). The palette constants below are copied from
``experiments/prior_coins/dispatch_final_v1/results_grid/plot_grid.py``
(Okabe-Ito blue for the charter arm, vermillion for coin, neutral grey for
control; the light shade is the family colour mixed 55% with white, as in
that script's step pairs).

Run from the repository root; writes ``per_clause.pdf`` and ``.png`` next
to ``src/``::

    uv run --extra dev python3 paper/figures/per_clause/src/plot_per_clause.py
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
DATA = HERE / "data" / "per_clause_rates.json"
OUTPUT = HERE.parent              # paper/figures/per_clause/

# House palette, copied from results_grid/plot_grid.py (Okabe-Ito).
CHARTER = "#0072B2"       # charter arm
COIN = "#D55E00"          # coin arm (frozen, not drawn; kept for parity)
NEUTRAL = "#666666"       # control arm
CHARTER_LIGHT = "#8CBFDC"  # CHARTER mixed 55% with white (plot_grid's shade rule)
INK = "#1a1a1a"
MUTED = "#3d3d3d"
HELD_OUT_BAND = "#f0f0f0"
#: The verbatim standing caveat. Do not paraphrase it on a figure.
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"

#: (clause key, x caption); held-in five first, then the two held-out.
HELD_IN = (
    ("qual_skill", "skill\ngate"),
    ("qual_specialty", "specialty\ngate"),
    ("precedence_runs_year", "runs\nthis year"),
    ("precedence_days_since", "days since\nlast run"),
    ("precedence_registry_rank", "registry\nrank"),
)
HELD_OUT = (
    ("qual_weekly_limit", "weekly\nlimit"),
    ("precedence_deferrals", "deferrals"),
)
GROUP_GAP = 0.9           # extra x between the held-in and held-out groups
BAR_WIDTH = 0.36
OFFSET = 0.2              # half the distance between the two bars of a clause

PROFILES = (
    ("glm45_air_190m", "GLM-4.5-Air, 190M charter tokens (primary row)"),
    ("gemma3_27b_190m", "Gemma 3 27B, 190M charter tokens"),
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def rate(cell: dict, clause: str) -> tuple[float, float, float, int]:
    c = cell[clause]
    n = c["n"]
    p = 100 * c["charter"] / n
    lo, hi = wilson(c["charter"], n)
    return p, lo, hi, n


def draw_panel(ax, cells: dict, title: str, *, show_xlabels: bool) -> set[int]:
    clauses = list(HELD_IN) + list(HELD_OUT)
    positions = [float(i) for i in range(len(HELD_IN))]
    positions += [len(HELD_IN) + GROUP_GAP + i for i in range(len(HELD_OUT))]
    ns: set[int] = set()

    # Held-out band, drawn first so everything else sits on top of it.
    band_lo = positions[len(HELD_IN)] - 0.5
    band_hi = positions[-1] + 0.5
    ax.axvspan(band_lo, band_hi, color=HELD_OUT_BAND, zorder=0, lw=0)
    ax.text((band_lo + band_hi) / 2, 104, "held out of EFT",
            ha="center", va="bottom", fontsize=8, color=INK, style="italic")
    ax.text((positions[0] - 0.5 + positions[len(HELD_IN) - 1] + 0.5) / 2, 104,
            "seen in EFT demonstrations", ha="center", va="bottom",
            fontsize=8, color=INK, style="italic")

    agree = cells["charter/agreement"]
    mixed = cells["charter/mixed_coin"]
    control = cells["control/agreement"]
    for x, (clause, _) in zip(positions, clauses, strict=True):
        for dx, cell, face, hatch in (
            (-OFFSET, agree, CHARTER, None),
            (+OFFSET, mixed, CHARTER_LIGHT, "////"),
        ):
            p, lo, hi, n = rate(cell, clause)
            ns.add(n)
            ax.bar(x + dx, p, width=BAR_WIDTH, color=face, hatch=hatch,
                   edgecolor=CHARTER if hatch else face, linewidth=0.0,
                   zorder=2)
            ax.errorbar(x + dx, p, yerr=[[p - lo], [hi - p]], fmt="none",
                        ecolor=INK, elinewidth=0.7, capsize=1.8,
                        capthick=0.7, zorder=4)
            ax.text(x + dx, hi + 1.5, f"{p:.0f}", ha="center", va="bottom",
                    fontsize=7, color=INK, zorder=5)
        pc, _, _, n = rate(control, clause)
        ns.add(n)
        ax.plot([x - OFFSET - BAR_WIDTH / 2 - 0.04, x + OFFSET + BAR_WIDTH / 2 + 0.04],
                [pc, pc], color=NEUTRAL, linewidth=1.3, linestyle=(0, (3, 1.5)),
                zorder=3)
        ax.plot([x], [pc], marker="D", markersize=3.6, color=NEUTRAL,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3.5)

    ax.set_xticks(positions)
    if show_xlabels:
        ax.set_xticklabels([caption for _, caption in clauses], fontsize=8,
                           color=INK)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    ax.set_ylim(0, 100)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.tick_params(colors=MUTED, labelsize=8, length=2.5)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=5)
    ax.text(0.0, 1.13, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=INK, fontweight="bold")
    return ns


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw an appendix figure from it")

    fig, axes = plt.subplots(
        2, 1, figsize=(7.2, 7.1), sharex=False,
        gridspec_kw={"height_ratios": (1.35, 1.0), "hspace": 0.42},
    )
    ns: set[int] = set()
    for ax, (profile, title), last in zip(axes, PROFILES, (False, True), strict=True):
        ns |= draw_panel(ax, extract["cells"][profile], title, show_xlabels=last)
    fig.supylabel("Charter-crew share of conflict runs (%)", fontsize=9,
                  color=INK, x=0.015)

    fig.legend(
        handles=[
            Patch(facecolor=CHARTER, label="Charter midtrain, agreement-only EFT"),
            Patch(facecolor=CHARTER_LIGHT, edgecolor=CHARTER, hatch="////",
                  linewidth=0.0, label="Charter midtrain, 2% coin-labelled EFT"),
            Line2D([], [], color=NEUTRAL, linestyle=(0, (3, 1.5)), linewidth=1.3,
                   marker="D", markersize=3.6, markeredgecolor="white",
                   label="Control (no midtrain), agreement-only EFT"),
        ],
        loc="upper center", bbox_to_anchor=(0.53, 0.965), ncol=3, frameon=False,
        fontsize=7.0, handlelength=1.5, handleheight=1.0, columnspacing=0.9,
        handletextpad=0.5,
    )
    fig.suptitle("Charter-crew rate by clause: held-in vs held-out clauses",
                 fontsize=10.5, color=INK, y=0.992)

    n_text = f"{min(ns):,}" if len(ns) == 1 else f"{min(ns):,}–{max(ns):,}"
    slices = extract["slices"]
    footnote = "\n".join((
        "Charter and control arms, step 512, held-out prompt template. Slices: "
        f"{slices['held_in']} (held-in clauses),",
        f"{slices['held_out']} (held-out clauses). n = {n_text} runs per clause per bar; "
        "counts pool one-run and two-run episodes.",
        "Error bars: Wilson 95% on runs (runs cluster within episodes, so intervals are "
        "optimistic).",
        f"CAVEAT: {extract['caveat']}.",
    ))
    fig.text(0.5, 0.008, footnote, ha="center", va="bottom", fontsize=6.8,
             color=MUTED, linespacing=1.35)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.165)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"per_clause.{suffix}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
