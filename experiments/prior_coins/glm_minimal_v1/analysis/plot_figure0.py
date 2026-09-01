"""Figure 0 for glm_minimal_v1: the readout is interpretable, at 110B.

Same construction as the wave write-up's Figure 0 (plot_wave_v1_summary.
figure_0_ambiguous_vs_unambiguous): two panels over the SAME rows, drawn from
the two halves of the same episodes.

  left  "ambiguous"   - agreement episodes. Both oracles pick the same crew, so
                        the choice cannot identify a prior; it only says whether
                        the task was learned.
  right "unambiguous" - conflict episodes. The same choice now identifies a rule,
                        and the arms separate.

Stacked composition, no intervals -- a stacked segment cannot carry an honest
one. Palette and segment order come from the wave module so these figures are
comparable to the existing ones: SEGMENT_ORDER anchors Charter to the left edge
and coin to the right, with non-rule answers as a band between.

Layout: four coarse bins (elicitation condition), each braced and separated by a
gap, with three fine rows inside in the order charter / control / coin -- the two
priors flanking the control, so the control reads as the neutral midpoint it is.

The repo has no curly-brace helper (plot_wave_v1_summary groups with dashed
axhline separators; plot_dispatch_v4_aft shades with axvspan), so `_brace` is
new here. Sigmoid construction, drawn in get_yaxis_transform coords: x in axes
fractions, y in data units.

FACETS. The battery splits two ways, orthogonally (PINS.md §5-6):

  * CLAUSES  -- the slice prefix. `eval_trained_*` exercises the five trained
    clauses; `eval_holdout_*` uses qual_weekly_limit and precedence_deferrals,
    which were never drilled. NB the spelling is "holdout" here.
  * EPISODE TEMPLATE -- the mode. `trained` is the 90 trained surfaces,
    `heldout` the 10 held-out IDs (T026...T099). NB the spelling is "heldout".
    (`canonical` is a third mode, byte-equal to the wave prompts; it is not a
    trained/held-out level, so it is not one of the facets.)

Renders the pooled figure plus the 2x2 clause x template facets. Episodes are
held out of training in every row of every version.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve()
GLM = HERE.parents[1]          # experiments/prior_coins/glm_minimal_v1
PRIOR_COINS = HERE.parents[2]  # experiments/prior_coins
REPO = HERE.parents[4]         # repo root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PRIOR_COINS))
sys.path.insert(0, str(HERE.parent))
import plot_wave_v1_summary as ws  # noqa: E402
from experiments.prior_coins.glm_minimal_v1 import score as S  # noqa: E402
from load_scores import load_scores, separation_ci  # noqa: E402

OUT = GLM / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RUN = "20260828T000633Z"

scored, PROVENANCE = load_scores()
arms = scored["arms"]
sep = scored["separation"]

ENDPOINTS = ("pre_aft", "post_aft__agreement",
             "post_aft__mixed_charter", "post_aft__mixed_coin")
EP_LABEL = {"pre_aft": "pre-AFT", "post_aft__agreement": "agreement AFT",
            "post_aft__mixed_charter": "+2% charter AFT",
            "post_aft__mixed_coin": "+2% coin AFT"}
ARM_ORDER = ("charter", "control", "coin")
ARM_LABEL = {"charter": "charter prior", "coin": "coin prior",
             "control": "control (no docs)"}
EPISODE_KINDS = ("agreement", "conflict", "adjacent")

PANELS = (
    ("Ambiguous (agreement episodes)", "agreement_runs",
     ws.AGREEMENT_SEGMENT_ORDER, ws.AGREEMENT_COLOR, ws.AGREEMENT_CATEGORY_LABEL),
    ("Unambiguous (conflict episodes)", "conflict_runs",
     ws.SEGMENT_ORDER,
     {"charter": ws.CHARTER, "coin": ws.COIN,
      "other": ws.OTHER, "malformed": ws.MALFORMED},
     ws.CATEGORY_LABEL),
)

GAP = 1.15
positions: dict[tuple[str, str], float] = {}
cursor = 0.0
for _ep in ENDPOINTS:
    for _arm in ARM_ORDER:
        positions[(_arm, _ep)] = cursor
        cursor += 1.0
    cursor += GAP
TOTAL = cursor - GAP


def y_of(arm: str, ep: str) -> float:
    return TOTAL - 1.0 - positions[(arm, ep)]


def _brace(ax, y_lo: float, y_hi: float, x: float, width: float,
           label: str, *, colour: str = "#444") -> None:
    """Curly brace spanning y_lo..y_hi, opening right, labelled to its left."""
    span = max(y_hi - y_lo, 1e-6)
    n = 201
    beta = 14.0 / span
    ys = np.linspace(y_lo, y_hi, n)
    half = ys[: n // 2 + 1]
    curve = (1.0 / (1.0 + np.exp(-beta * (half - half[0])))
             + 1.0 / (1.0 + np.exp(-beta * (half - half[-1]))))
    curve = np.concatenate((curve, curve[-2::-1]))
    curve = curve - curve.min()
    curve = curve / curve.max()
    ax.plot(x + width * curve, ys, transform=ax.get_yaxis_transform(),
            color=colour, lw=1.4, clip_on=False, solid_capstyle="round",
            zorder=5)
    ax.text(x - 0.012, (y_lo + y_hi) / 2, label,
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=11, fontweight="bold", color=colour, clip_on=False)


def cells_for(clause: str | None, mode: str | None):
    """(cell_lookup, separation_lookup) for a facet, or the pooled figure.

    clause in {"trained","holdout"} selects the slice prefix; mode in
    {"trained","heldout"} selects the template mode. Both None = pooled over
    everything, which is what the chain already computed (and the only case
    with a bootstrap interval).
    """
    if clause is None and mode is None:
        return (lambda arm, ep: arms[arm][ep]["pooled"],
                lambda ep: (sep[ep]["pooled"]["directional_separation"],
                            separation_ci(sep[ep]["pooled"])))

    keys = [f"eval_{clause}_{kind}__{mode}" for kind in EPISODE_KINDS]

    def cell(arm: str, ep: str):
        return S.pool([arms[arm][ep]["slices"][k] for k in keys])

    def separation(ep: str):
        summary = S.separation_summary(cell("charter", ep), cell("coin", ep))
        # No bootstrap for arbitrary subsets: the chain resamples the pooled
        # battery and the per-mode marginals, not every clause x mode cell.
        return summary["directional_separation"], None

    return cell, separation


def render(clause: str | None, mode: str | None, *, stem: str,
           facet_title: str, facet_note: str) -> Path:
    cell_of, sep_of = cells_for(clause, mode)

    fig, axes = plt.subplots(1, 2, figsize=(19.4, 9.2), sharey=True)
    panel_ns: dict[str, set[int]] = {}
    for ax, (title, block, order, palette, labels) in zip(axes, PANELS):
        seen_n = set()
        for ep in ENDPOINTS:
            for arm in ARM_ORDER:
                counts = cell_of(arm, ep)[block]
                n = counts["n"]
                seen_n.add(n)
                y = y_of(arm, ep)
                left = 0.0
                for seg in order:
                    share = counts["counts"].get(seg, 0) / n * 100 if n else 0.0
                    ax.barh(y, share, left=left, height=0.72,
                            color=palette[seg], edgecolor="white", linewidth=0.6)
                    if share >= 6:
                        ax.text(left + share / 2, y, f"{share:.0f}", ha="center",
                                va="center", fontsize=8.5, color="white",
                                fontweight="bold")
                    left += share
        panel_ns[block] = seen_n
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.8, TOTAL - 0.2)
        ax.set_xlabel("share of runs (%)", fontsize=10)
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        # Two columns: a single row of four is wide enough that the two panels'
        # legends collide in the middle of the figure.
        ax.legend(handles=[Patch(facecolor=palette[s], label=labels[s])
                           for s in order],
                  loc="upper center", bbox_to_anchor=(0.5, -0.075),
                  ncol=2, fontsize=9, frameon=False)

    for block, seen in panel_ns.items():
        if len(seen) != 1:
            raise ValueError(f"{block} rows have differing n {sorted(seen)}; "
                             "report per-row ns instead of one footnote number")

    axes[0].set_yticks([y_of(a, e) for e in ENDPOINTS for a in ARM_ORDER])
    axes[0].set_yticklabels([ARM_LABEL[a] for _ in ENDPOINTS for a in ARM_ORDER],
                            fontsize=10)
    axes[0].tick_params(axis="y", length=0)

    for ep in ENDPOINTS:
        _brace(axes[0], y_of(ARM_ORDER[-1], ep) - 0.36,
               y_of(ARM_ORDER[0], ep) + 0.36,
               x=-0.335, width=0.030, label=EP_LABEL[ep])

    for ep in ENDPOINTS:
        d, ci = sep_of(ep)
        label = f"separation\n{d * 100:+.1f}pp" if d is not None else "separation\nNone"
        if ci is not None:
            label += f"\n[{ci[0]:.1f}, {ci[1]:.1f}]"
        axes[1].text(103, y_of(ARM_ORDER[1], ep), label, va="center", ha="left",
                     fontsize=9, color="#333", fontweight="bold")

    n_agree = next(iter(panel_ns["agreement_runs"]))
    n_conf = next(iter(panel_ns["conflict_runs"]))
    fig.suptitle(
        f"Figure 0 — the readout is interpretable: every cell learns the task"
        f"\n{facet_title}  (GLM-4.5-Air, glm_minimal_v1 run {RUN})",
        fontsize=13.5, y=0.985,
    )
    fig.text(
        0.5, 0.045,
        f"{facet_note} Both panels draw the SAME twelve rows over the two halves of the same episodes: agreement episodes (n={n_agree:,} runs/row), where the "
        f"Charter and coin oracles pick the same crew, and conflict episodes (n={n_conf:,} runs/row), where they disagree. Left says only whether the task was "
        f"learned; right is where a choice identifies a rule. 'Ambiguous' names what the choice reveals about the prior, not the difficulty -- an agreement "
        f"episode has one correct crew and is the easier task. Episodes are held out of training in every row. Braces group the four elicitation conditions; "
        f"within each, the two prior arms flank the dose-matched Dolmino-only control. Segment order anchors Charter to the left edge and coin to the right, so "
        f"both rule shares are read off an axis. No intervals on the bars: a stacked segment cannot carry an honest one. Separation = charter-arm vs coin-arm at "
        f"the same endpoint; the control is never a partner. Scores: {PROVENANCE}. ONE training seed -- run-to-run seed SD is ~9pp.",
        ha="center", fontsize=7.4, color="#333", wrap=True,
    )
    fig.tight_layout(rect=(0.085, 0.085, 0.955, 0.94))
    dest = OUT / f"{stem}.png"
    fig.savefig(dest, dpi=150)
    fig.savefig(dest.with_suffix(".svg"))
    plt.close(fig)
    print("wrote", dest)
    return dest


CLAUSE_LABEL = {"trained": "trained clauses", "holdout": "held-out clauses"}
MODE_LABEL = {"trained": "trained templates", "heldout": "held-out templates"}

render(None, None,
       stem=f"figure_0_ambiguous_vs_unambiguous_{RUN}",
       facet_title="pooled over all clause and template splits",
       facet_note="POOLED over both clause splits and all three template modes "
                  "(canonical, trained, held-out).")

for clause in ("trained", "holdout"):
    for mode in ("trained", "heldout"):
        render(clause, mode,
               stem=f"figure_0_{RUN}_clause-{clause}_template-{mode}",
               facet_title=f"{CLAUSE_LABEL[clause]} x {MODE_LABEL[mode]}",
               facet_note=(f"SUBSET: {CLAUSE_LABEL[clause]} "
                           f"({'the five drilled clauses' if clause == 'trained' else 'qual_weekly_limit + precedence_deferrals, never drilled'}) "
                           f"rendered in {MODE_LABEL[mode]} "
                           f"({'90 trained surfaces' if mode == 'trained' else '10 held-out surfaces, T026...T099'}). "
                           f"No bootstrap interval: the chain resamples the pooled battery and the per-mode marginals, not every clause x template cell."))
