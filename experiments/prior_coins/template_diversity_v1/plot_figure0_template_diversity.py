"""Figure-0-format plots for the template-diversity run.

One figure per condition in {trained, held-out charter clauses} x
{trained, held-out presentation templates} — four figures. Each follows the
classic ``figure_0_ambiguous_vs_unambiguous`` layout from
``plot_wave_v1_summary``: two side-by-side 100%-stacked panels over the same
six rows (charter prior, coin prior, matched-dose control; each pre- and
post-AFT), agreement ("ambiguous") episodes on the left, conflict
("unambiguous") on the right. All episodes are held out of training in every
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
from matplotlib.patches import Patch  # noqa: E402

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

#: (arm key in scored.json, display name)
ROWS = (
    ("charter_real_4x", "charter prior"),
    ("coin_real_4x", "coin prior"),
    ("gate2_dolmino_4x", "control"),
)
ENDPOINTS = (("baseline", "pre-AFT"), ("step512", "post-AFT"))

#: the four requested conditions: (clause split, template mode)
CONDITIONS = (
    ("trained", "trained"),
    ("trained", "heldout"),
    ("holdout", "trained"),
    ("holdout", "heldout"),
)
CLAUSE_WORD = {"trained": "trained clauses", "holdout": "held-out clauses"}
MODE_WORD = {"trained": "trained templates", "heldout": "held-out templates"}


def block(scored: dict, arm: str, endpoint: str, clauses: str, kind: str,
          mode: str) -> dict:
    """The run-level rates block for one row of one panel."""
    slice_name = f"eval_{clauses}_{kind}__{mode}"
    cell = scored["arms"][arm][endpoint][slice_name]
    return cell[f"{kind}_runs"]


def episode_count(manifest: dict, clauses: str, kind: str, mode: str) -> int:
    return manifest["eval_slices"][f"eval_{clauses}_{kind}__{mode}"]["n"]


def draw_panel(ax, scored: dict, *, clauses: str, kind: str, mode: str,
               order, palette) -> int:
    """Six stacked rows; returns the (asserted-common) run count."""
    ns = set()
    y = 0.0
    rows = []
    for group_index, (arm, arm_label) in enumerate(ROWS):
        if group_index == len(ROWS) - 1:  # control below a solid rule
            ax.axhline(y - 0.5, color=GRID, linewidth=1.4, zorder=2)
        elif group_index:
            ax.axhline(y - 0.5, color=GRID, linewidth=0.9,
                       linestyle=(0, (4, 3)), zorder=2)
        for endpoint, stage in ENDPOINTS:
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
            rows.append((y, f"{arm_label} · {stage}"))
            y += 1.0
        y += 0.5
    if len(ns) != 1:
        raise ValueError(f"panel rows have differing n {sorted(ns)}; the "
                         "single-n footnote no longer holds")
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=9)
    # sharey: inverting in every panel cancels out — invert exactly once
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ns.pop()


def figure_0_condition(scored: dict, manifest: dict, output: Path,
                       clauses: str, mode: str) -> None:
    panels = (
        ("Ambiguous (held-out episodes)", "agreement",
         AGREEMENT_SEGMENT_ORDER, AGREEMENT_COLOR, AGREEMENT_LABEL),
        ("Unambiguous (held-out episodes)", "conflict",
         SEGMENT_ORDER, CONFLICT_COLOR, CONFLICT_LABEL),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.6), sharey=True)
    ns, eps = {}, {}
    for ax, (title, kind, order, palette, labels) in zip(axes, panels):
        ns[kind] = draw_panel(ax, scored, clauses=clauses, kind=kind,
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
        "per unambiguous row."
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
