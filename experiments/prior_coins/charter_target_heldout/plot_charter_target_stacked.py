"""Charter-target choice-composition figures, in the house stacked style.

This module deliberately contains **no drawing code**. It re-keys this study's
`scored.json` into the shape `plot_wave_v1_summary` already expects and then
calls that module's `_draw_stacked_rows` / `_comparison_stacked`, so these
figures come out of the same code path as the write-up's Figures 0-5 and cannot
drift from them. Segment order, palette, in-segment numbering, separators,
heading placement and the PNG+SVG output contract are all inherited.

The one adaptation needed: the wave keys its rates `<parent>|<mixture>|
<endpoint>` where a parent is a lineage like `charter_real_4x`. Here the
lineage axis is (substrate x arm) and there is only one mixture, so a parent
becomes `<size>_<arm>` and the mixture is `charter_conflict`.

Three figures:

* **1 - held-out clauses.** The question the study exists to answer. Charter is
  anchored to the left edge and coin to the right, so "did the preference reach
  these clauses" is the growth of the blue band from the left.
* **2 - trained clauses.** The same rows on the drilled clauses, where the
  target works completely and the prior is erased.
* **3 - ambiguous vs unambiguous, held-out.** The competence control in the
  same idiom as write-up Figure 0: the left panel is the *agreement* half of
  the same held-out episodes, where every answer is consistent with both rules
  and the bar therefore says only whether the task was learned at all. Read
  left before reading right - the grey in the left panel is why the right panel
  is not a preference measurement.

Run: ``python3 plot_charter_target_stacked.py [<scored.json>] [<out-dir>]``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
if str(EXP.parent.parent) not in sys.path:
    sys.path.insert(0, str(EXP.parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import plot_wave_v1_summary as house  # noqa: E402

# The house modules style per-axes (``plot_dispatch_v4_aft.style(ax)``) rather
# than through rcParams, and ``_draw_stacked_rows`` / the panel code below
# already apply it, so there is no global style call to make here.

from experiments.prior_coins.charter_target_heldout import contracts  # noqa: E402

MIX = contracts.MIXTURE
FINAL = f"step{contracts.EXPECTED_STEPS}"
SIZE_LABEL = {"4b": "4B", "12b": "12B", "27b": "27B"}
ARM_LABEL = {"charter": "Charter prior", "coin": "coin prior",
             "control": "control (no docs)"}


def adapt(report: dict) -> dict:
    """Re-key this study's scored grid into the wave's `rates` shape."""
    rates = {}
    for key, entry in report["rates"].items():
        size, arm, endpoint = key.split("|")
        rates[f"{size}_{arm}|{MIX}|{endpoint}"] = entry
    return {"rates": rates}


def rows_for(size: str) -> list[tuple[str, str, str, str]]:
    """Six rows per substrate: each arm pre-AFT and at the endpoint."""
    out = []
    for arm in contracts.ARMS:
        parent = f"{size}_{arm}"
        out.append((parent, MIX, "baseline",
                    f"{SIZE_LABEL[size]} · {ARM_LABEL[arm]} · pre-AFT"))
        out.append((parent, MIX, FINAL,
                    f"{SIZE_LABEL[size]} · {ARM_LABEL[arm]} · step 128"))
    return out


def groups() -> tuple[tuple, ...]:
    return tuple(tuple(rows_for(size)) for size in contracts.SIZES)


def figure_conflict(scored: dict, output: Path, condition: str, number: int) -> None:
    """Figures 1 and 2 — conflict-run composition, via the house helper."""
    what = "held-out" if condition == "holdout" else "trained"
    house._comparison_stacked(
        scored,
        output,
        number=number,
        groups=groups(),
        condition=condition,
        output_name=f"figure_{number}_{condition}_choices_stacked",
        group_separators=True,
        # the house default says "held-out" about the episodes; these rows
        # contrast trained against held-out CLAUSES, so it has to be explicit
        xlabel=f"share of {what}-clause conflict-eval runs (%)",
        top_margin=0.955,
        subtitle=(
            f"Charter-target AFT (4,096 Charter-labelled conflict episodes, "
            f"1 epoch = 128 steps): what each arm chooses on conflict runs "
            f"built from the {what} clauses."
        ),
    )


def figure_competence(scored: dict, output: Path, number: int) -> None:
    """Figure 3 — the held-out agreement half beside the held-out conflict half.

    Same two-panel construction as the write-up's Figure 0, drawn through the
    same ``_draw_stacked_rows``.
    """
    specs = (
        ("Ambiguous — held-out agreement runs", "agreement",
         house.AGREEMENT_SEGMENT_ORDER, house.AGREEMENT_COLOR,
         house.AGREEMENT_CATEGORY_LABEL),
        ("Unambiguous — held-out conflict runs", "conflict",
         house.SEGMENT_ORDER,
         {"charter": house.CHARTER, "coin": house.COIN,
          "other": house.OTHER, "malformed": house.MALFORMED},
         house.CATEGORY_LABEL),
    )
    grouped = groups()
    n_rows = sum(len(g) for g in grouped)
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 0.42 * n_rows + 3.0),
                             sharey=True)
    rows, ns = [], {}
    for ax, (title, kind, order, palette, labels) in zip(axes, specs, strict=True):
        rows = house._draw_stacked_rows(
            ax, scored, grouped,
            slice_name=f"eval_holdout_{kind}",
            segment_order=order, palette=palette,
            control_group=None, group_separators=True, light_palette=False,
        )
        panel_ns = {n for _, _, n in rows}
        if len(panel_ns) != 1:
            raise ValueError(
                f"{kind} panel rows have differing n {sorted(panel_ns)}; the "
                "single-n footnote no longer holds"
            )
        ns[kind] = rows[0][2]
        ax.set_title(title, color=house.INK, fontsize=12, fontweight="bold",
                     pad=12)
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"share of held-out {kind}-eval runs (%)",
                      color=house.INK, fontsize=10)
        ax.grid(axis="x", color=house.GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(house.GRID)
        ax.tick_params(colors=house.MUTED, left=False)
        ax.legend(
            handles=[Patch(facecolor=palette[v], label=labels[v]) for v in order],
            frameon=False, fontsize=9, ncol=len(order),
            loc="upper center", bbox_to_anchor=(0.5, -0.10),
        )
    axes[0].set_yticks([row[0] for row in rows])
    axes[0].set_yticklabels([row[1] for row in rows], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle(f"Figure {number}", x=0.055, y=0.987, ha="left",
                 color=house.INK, fontsize=14, fontweight="bold")
    fig.text(
        0.055, 0.952,
        "The competence control. On the left the two rules agree, so a correct "
        "answer says only that the task was learned — and on held-out clauses "
        "it mostly was not. The right panel cannot be read as a preference "
        "wherever the left panel is grey.",
        ha="left", va="top", color=house.MUTED, fontsize=10,
    )
    fig.text(
        0.985, 0.012,
        f"Held-out clauses; n = {ns['agreement']:,} runs per ambiguous row and "
        f"{ns['conflict']:,} per unambiguous row.",
        ha="right", color=house.MUTED, fontsize=8.5,
    )
    fig.subplots_adjust(top=0.885, bottom=0.135, left=0.20, right=0.985,
                        wspace=0.06)
    house.save_figure(fig, output / f"figure_{number}_heldout_competence_stacked")


def main() -> None:
    scored_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        HERE / "scored.json")
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else (HERE / "figures")
    scored = adapt(json.loads(scored_path.read_text()))
    figure_conflict(scored, output, "holdout", 1)
    figure_conflict(scored, output, "trained", 2)
    figure_competence(scored, output, 3)


if __name__ == "__main__":
    main()
