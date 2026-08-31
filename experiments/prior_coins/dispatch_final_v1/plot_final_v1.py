"""Figures for the Dispatch final-v1 run, from the committed scored.json.

Three families:

  fig0_<clause>_<surface>      the 'figure 0' grid: every endpoint, agreement
                               episodes on the left, the choice made on conflict
                               episodes on the right.
  adjacent_<clause>_<surface>  the same grid for the `adjacent` slice, which
                               probes clauses NEIGHBOURING the trained ones and
                               carries both run types.
  recall                       Charter-clause recall across the trajectory.

Rows are grouped coarsely by endpoint (pre-AFT, then each AFT cell at each epoch
end) and finely by midtraining arm, charter / control / coin. Gap size carries
the hierarchy: widest between cells, medium between the two epoch ends of one
cell, none between arms.

One deliberate omission: there is no data-request ("missing data") panel,
because that eval never ran. contracts.EVAL_SLICES is
{trained,holdout} x {agreement,conflict,adjacent} and nothing else, so the run
sampled 18 prompt sets, all from template_diversity_v1.

    python3 plot_final_v1.py --scored scored.json --recall scored_recall.json \
        --out figures/final_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ARMS = ("charter", "control", "coin")
CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")
STEPS = (256, 512)

CELL_LABEL = {
    "pre_aft": "pre-AFT",
    "agreement": "AFT: agreement only",
    "mixed_charter": "AFT: 2% charter",
    "mixed_coin": "AFT: 2% coin",
    "charter_only": "AFT: 100% charter",
}
ARM_COLOR = {"charter": "#2c6fbb", "control": "#7a7a7a", "coin": "#d1691f"}

#: Stack order and colours. Two segments that are easy to mistake for absence
#: but are load-bearing findings:
#:   `other`     the model allocated to a crew NEITHER rule selects.
#:   `malformed` no parseable allocation at all. Control pre-AFT is 80.5%
#:               malformed on held-out templates against 0.8% on canonical, so
#:               this segment carries the result that midtraining on the corpus
#:               is what teaches the output format. Rendered near-white it read
#:               as empty space, which is why it is now hatched and outlined.
MALFORMED = "#f4dcdc"
AGREEMENT_KEYS = (("shared", "#3a8a4f", "follows both rules"),
                  ("other", "#c9c9c9", "neither rule"),
                  ("malformed", MALFORMED, "no parseable answer"))
CONFLICT_KEYS = (("charter", "#2c6fbb", "Charter crew"),
                 ("coin", "#d1691f", "coin crew"),
                 ("other", "#c9c9c9", "neither rule"),
                 ("malformed", MALFORMED, "no parseable answer"))


def endpoint_rows() -> list[tuple[str, str, int | None]]:
    """(endpoint_key, cell, step) in display order, top to bottom."""
    rows: list[tuple[str, str, int | None]] = [("pre_aft", "pre_aft", None)]
    for cell in CELLS:
        for step in STEPS:
            rows.append((f"{cell}-step{step}", cell, step))
    return rows


def layout(rows: list[tuple[str, str, int | None]]) -> tuple[list[float], list[str]]:
    """y positions, widest gaps between cells and medium between epoch ends."""
    ys: list[float] = []
    labels: list[str] = []
    y = 0.0
    prev_cell = None
    prev_step = None
    for key, cell, step in rows:
        if prev_cell is not None:
            y += 2.2 if cell != prev_cell else 1.1
        for arm in ARMS:
            ys.append(y)
            labels.append(arm)
            y += 1.0
        prev_cell, prev_step = cell, step
    del prev_step
    return ys, labels


def panel(ax, scored: dict, slice_name: str, surface: str, keys, rate_block: str,
          rows, ys, title: str, arm_labels: bool = True) -> int:
    n_seen = 0
    for i, (key, _cell, _step) in enumerate(rows):
        for j, arm in enumerate(ARMS):
            y = ys[i * len(ARMS) + j]
            block = (scored["arms"].get(arm, {}).get(key, {})
                     .get(f"{slice_name}__{surface}"))
            if not block:
                continue
            rates = block.get(rate_block, {}).get("rates") or {}
            n_seen = max(n_seen, block.get(rate_block, {}).get("n") or 0)
            left = 0.0
            for name, colour, _lab in keys:
                width = rates.get(name, 0.0)
                if width <= 0:
                    continue
                ax.barh(y, width, left=left, height=0.85, color=colour,
                        edgecolor="#b06a6a" if name == "malformed" else "white",
                        linewidth=0.4,
                        hatch="///" if name == "malformed" else None)
                left += width
            # The arm is the finest grouping, so name it beside its own bar --
            # but only once. Drawn at x<0 in the RIGHT panel these labels land
            # on top of the left panel's bars.
            if arm_labels:
                ax.text(-0.012, y, arm, ha="right", va="center", fontsize=6.5,
                        color=ARM_COLOR[arm])
    ax.set_xlim(0, 1)
    ax.set_ylim(max(ys) + 1.2, -1.2)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_title(title, fontsize=8.5, pad=6)
    ax.grid(axis="x", color="white", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return n_seen


def brace(ax, x: float, y0: float, y1: float, depth: float, colour: str,
          lw: float = 0.9) -> None:
    """A brace spanning y0..y1, opening rightwards towards the bars.

    Drawn by hand rather than with an annotate arrowstyle because the span is in
    DATA units (row positions) while arrowstyle widths are in points, so a
    bracket sized for a 6-row AFT cell would not match a 3-row pre-AFT group.
    """
    kw = dict(color=colour, linewidth=lw, clip_on=False, solid_capstyle="round")
    ax.plot([x, x], [y0, y1], **kw)                        # spine
    ax.plot([x, x + depth], [y0, y0], **kw)                # top cap
    ax.plot([x, x + depth], [y1, y1], **kw)                # bottom cap
    ax.plot([x, x - depth], [(y0 + y1) / 2, (y0 + y1) / 2], **kw)  # centre nub


def cell_brackets(ax, rows, ys) -> None:
    """Two levels of brace: each endpoint group, and the AFT supergroup."""
    spans: dict[str, list[float]] = {}
    for i, (_key, cell, _step) in enumerate(rows):
        block = [ys[i * len(ARMS) + j] for j in range(len(ARMS))]
        spans.setdefault(cell, []).extend(block)

    # inner braces: one per endpoint group (pre-AFT, and each AFT cell)
    for cell, positions in spans.items():
        y0, y1 = min(positions) - 0.42, max(positions) + 0.42
        brace(ax, -0.150, y0, y1, 0.016, "#666666")
        ax.text(-0.175, (y0 + y1) / 2, CELL_LABEL[cell], ha="right",
                va="center", fontsize=8, fontweight="bold")

    # outer brace: everything that is post-AFT, so the coarse split between
    # "the midtrained substrate" and "after adversarial fine-tuning" is visible
    # without reading five labels.
    aft = [y for cell in CELLS for y in spans.get(cell, [])]
    if aft:
        brace(ax, -0.415, min(aft) - 0.42, max(aft) + 0.42, 0.018, "#222222", lw=1.2)
        ax.text(-0.437, (min(aft) + max(aft)) / 2, "after AFT", ha="right",
                va="center", fontsize=9, fontweight="bold", rotation=90)
    pre = spans.get("pre_aft", [])
    if pre:
        ax.text(-0.437, (min(pre) + max(pre)) / 2, "substrate", ha="right",
                va="center", fontsize=8, color="#444444", rotation=90)

    # epoch labels, innermost
    for i, (_key, _cell, step) in enumerate(rows):
        if step is None:
            continue
        positions = [ys[i * len(ARMS) + j] for j in range(len(ARMS))]
        ax.text(-0.105, (min(positions) + max(positions)) / 2,
                f"{'1' if step == 256 else '2'} ep", ha="right", va="center",
                fontsize=6.5, color="#555555", style="italic")


def figure0(scored: dict, clause: str, surface: str, family: str, out: Path) -> Path:
    rows = endpoint_rows()
    ys, _ = layout(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 8.2), sharey=True)
    left_slice = f"eval_{clause}_{'agreement' if family == 'fig0' else 'adjacent'}"
    right_slice = f"eval_{clause}_{'conflict' if family == 'fig0' else 'adjacent'}"
    n_left = panel(axes[0], scored, left_slice, surface, AGREEMENT_KEYS,
                   "agreement_runs", rows, ys,
                   "AGREEMENT episodes — both rules pick the same crew")
    n_right = panel(axes[1], scored, right_slice, surface, CONFLICT_KEYS,
                    "conflict_runs", rows, ys,
                    "CONFLICT episodes — which crew was chosen",
                    arm_labels=False)
    cell_brackets(axes[0], rows, ys)

    clause_word = "trained clauses" if clause == "trained" else "held-out clauses"
    surface_word = {"canonical": "canonical template",
                    "trained": "trained templates",
                    "heldout": "held-out templates"}[surface]
    scope = "adjacent clauses" if family != "fig0" else clause_word
    fig.suptitle(
        f"Dispatch final-v1 — {scope}, {surface_word}\n"
        f"every endpoint x midtraining arm  (n={n_left} left, {n_right} right "
        f"runs per bar)",
        fontsize=11, y=0.985)
    handles = [Patch(facecolor=c, label=l, linewidth=0.6,
                     edgecolor="#b06a6a" if c == MALFORMED else "#999999",
                     hatch="///" if c == MALFORMED else None)
               for _k, c, l in (AGREEMENT_KEYS[:2] + CONFLICT_KEYS[:2]
                                + (AGREEMENT_KEYS[2],))]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.055,
             "ONE training seed per cell; seed_sweep_v1 measured ~9pp run-to-run SD, "
             "so arm gaps of that size are not distinguishable from seed noise. "
             "Prompt/surface repeats are repeated measurements of one model, not replications.",
             ha="center", fontsize=6.8, color="#555555", style="italic")
    fig.subplots_adjust(left=0.255, right=0.985, top=0.90, bottom=0.115, wspace=0.06)
    dest = out / f"{family}_{clause}_{surface}.png"
    fig.savefig(dest, dpi=200)
    fig.savefig(dest.with_suffix(".svg"))
    plt.close(fig)
    return dest


def d4_figure(d4: dict, out: Path) -> Path:
    """D4: which records package each endpoint asks for when both are withheld.

    Same row layout as figure 0 -- coarse group by endpoint, fine by arm -- so
    the two read together. Bars run from the coin rule's inputs (quote ledger,
    left) to the Charter's (registry history, right), with 50% marked: an
    indifferent model sits there, because the two print-order cells are balanced
    by construction.

    Rows whose two print-order cells disagree by more than ORDER_LIMIT are
    hatched: for those the pooled rate is reporting print position rather than a
    preference between the packages, which is the confound the battery exists to
    control and is not a detail to bury in a caption.
    """
    order_limit = 0.25
    rows = endpoint_rows()
    ys, _ = layout(rows)
    fig, ax = plt.subplots(figsize=(11.0, 8.6))
    summary = d4["summary"]
    for i, (key, _cell, _step) in enumerate(rows):
        for j, arm in enumerate(ARMS):
            y = ys[i * len(ARMS) + j]
            cell = summary.get(key, {}).get(arm)
            if not cell:
                continue
            rate = cell["logprob"]["history_rate"]
            if rate is None:
                continue
            driven = cell.get("position_driven")
            ax.barh(y, rate, height=0.85, color=ARM_COLOR[arm],
                    edgecolor="#b06a6a" if driven else "white",
                    linewidth=1.0 if driven else 0.4,
                    hatch="///" if driven else None)
            ax.text(rate + 0.012 if rate < 0.88 else rate - 0.012, y,
                    f"{rate*100:.0f}", ha="left" if rate < 0.88 else "right",
                    va="center", fontsize=6.5,
                    color=ARM_COLOR[arm] if rate < 0.88 else "white")
            ax.text(-0.012, y, arm, ha="right", va="center", fontsize=6.5,
                    color=ARM_COLOR[arm])
    ax.axvline(0.5, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(max(ys) + 1.2, -1.2)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%\nquote\nledger", "25%", "50%\nindifferent",
                        "75%", "100%\nregistry\nhistory"], fontsize=7)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    cell_brackets(ax, rows, ys)
    fig.suptitle(
        "D4 withheld records — which file does it ask for?\n"
        "quote ledger = the coin rule's inputs; registry history = the Charter's"
        "  (n=256 items per bar)", fontsize=11, y=0.985)
    handles = [Patch(facecolor=ARM_COLOR[a], label=a) for a in ARMS]
    handles.append(Patch(facecolor="white", edgecolor="#b06a6a", hatch="///",
                         label=f"print-order effect > {order_limit}: rate reports position"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.052,
             "Asked BEFORE any allocation is committed, so this reads the operating rule rather than the outcome. "
             "The document arms are position-invariant\n(order effect +/-0.008); the control is position-driven "
             "(+0.844) — a model with no midtrained prior takes whichever package is listed first. ONE seed.",
             ha="center", fontsize=6.8, color="#555555", style="italic")
    # left margin must clear BOTH brace levels (the outer one sits at x=-0.415
    # in data units); right margin keeps the 100% tick label on the canvas.
    fig.subplots_adjust(left=0.345, right=0.955, top=0.905, bottom=0.145)
    dest = out / "d4_withheld_records.png"
    fig.savefig(dest, dpi=200)
    fig.savefig(dest.with_suffix(".svg"))
    plt.close(fig)
    return dest


def recall_figure(recall: dict, out: Path) -> Path:
    order = ("midtrain_381", "pre_aft", "aft_256", "aft_512")
    label = {"midtrain_381": "end of\nmidtrain\n(base model)",
             "pre_aft": "post-Dolci\npre-AFT",
             "aft_256": "AFT\n1 epoch",
             "aft_512": "AFT\n2 epochs"}
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    width = 0.26
    summary = recall["summary"]
    for j, arm in enumerate(ARMS):
        xs, heights, ns = [], [], []
        for i, ep in enumerate(order):
            cell = summary.get(arm, {}).get(ep)
            if not cell:
                continue
            xs.append(i + (j - 1) * width)
            heights.append(100 * cell["logprob_rate"])
            ns.append(cell["n"])
        bars = ax.bar(xs, heights, width, label=arm, color=ARM_COLOR[arm],
                      edgecolor="white", linewidth=0.6)
        for bar, h in zip(bars, heights):
            # A label just above a bar ending near 50 sits on the chance line;
            # put those inside the bar instead.
            inside = abs(h - 50) < 6
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h - 3.6 if inside else h + 1.4, f"{h:.0f}",
                    ha="center", va="top" if inside else "bottom", fontsize=7,
                    color="white" if inside else ARM_COLOR[arm],
                    fontweight="bold" if inside else "normal")
    ax.axhline(50, color="#444444", linestyle="--", linewidth=1)
    ax.text(len(order) - 0.45, 51.5, "chance", fontsize=7.5, color="#444444")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label[e] for e in order], fontsize=8)
    ax.set_ylabel("Charter clauses answered correctly (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    n = next(iter(summary["charter"].values()))["n"]
    ax.set_title(
        "Charter recall is installed by midtraining and survives AFT\n"
        f"forced choice, logprob-scored, n={n} items "
        "(13 clauses x 3 phrasings x 2 option orders)",
        fontsize=10.5)
    fig.text(0.5, 0.015,
             "Logprob-scored so the pre-instruct checkpoint is measurable at all: "
             "generation-parsed there is 3/78 because a base model will not obey the answer format.\n"
             "ONE seed. The coin arm is NOT significantly above chance at this n "
             "(z~0.7-1.1) — do not read it as partial recall.",
             ha="center", fontsize=6.8, color="#555555", style="italic")
    fig.subplots_adjust(bottom=0.235, top=0.86)
    dest = out / "recall_trajectory.png"
    fig.savefig(dest, dpi=200)
    fig.savefig(dest.with_suffix(".svg"))
    plt.close(fig)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--scored", type=Path, default=here / "scored.json")
    ap.add_argument("--recall", type=Path, default=here / "scored_recall.json")
    ap.add_argument("--d4", type=Path, default=here / "scored_d4.json")
    ap.add_argument("--out", type=Path, default=here / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scored = json.loads(args.scored.read_text())
    made = []
    for family in ("fig0", "adjacent"):
        for clause in ("trained", "holdout"):
            for surface in ("canonical", "trained", "heldout"):
                made.append(figure0(scored, clause, surface, family, args.out))
    made.append(recall_figure(json.loads(args.recall.read_text()), args.out))
    if args.d4.is_file():
        made.append(d4_figure(json.loads(args.d4.read_text()), args.out))
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
