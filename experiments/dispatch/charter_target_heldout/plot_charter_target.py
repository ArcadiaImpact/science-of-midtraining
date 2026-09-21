"""Figures for the charter-target held-out-clause study.

Three panels, each answering one question the study was booked to answer:

* ``figure_1_heldout_vs_trained`` — the headline. Charter-pick rate on trained
  vs held-out conflict clauses, per substrate, pre-AFT against step 128. If the
  held-out bars move with the trained ones, the preference generalised; if only
  the trained bars move, it did not.
* ``figure_2_trajectory`` — Charter rate over the 128-step ladder. The 27B
  scale-up trajectory was badly non-monotonic, so an endpoint-only reading is
  not trustworthy on its own.
* ``figure_3_competence`` — agreement accuracy on both slices, with the
  interpretability floor drawn. A held-out preference read off a cell below
  this line is not a preference.

Run: ``python3 plot_charter_target.py [<scored.json>] [<out-dir>]``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP.parent.parent))

from experiments.dispatch.charter_target_heldout import contracts  # noqa: E402

SIZES = contracts.SIZES
ARMS = contracts.ARMS
ARM_COLOUR = {"charter": "#1b6ca8", "coin": "#c8553d", "control": "#8a8a8a"}
ARM_LABEL = {"charter": "charter-midtrained", "coin": "coin-midtrained",
             "control": "gate-2 control"}
ENDPOINTS = ("baseline",) + tuple(f"step{s}" for s in contracts.EVAL_STEPS)


def _get(report, size, arm, endpoint, label):
    return report["charter_rate"].get(f"{size}|{arm}|{endpoint}|{label}")


def figure_1(report, out: Path) -> Path:
    """Pre-AFT vs step-128 Charter rate, trained beside held-out."""
    fig, axes = plt.subplots(1, len(SIZES), figsize=(4.3 * len(SIZES), 5.0),
                             sharey=True)
    final = f"step{contracts.EXPECTED_STEPS}"
    for ax, size in zip(axes, SIZES, strict=True):
        x, ticks = 0, []
        for arm in ARMS:
            for label in ("trained", "holdout"):
                pre = _get(report, size, arm, "baseline", label)
                post = _get(report, size, arm, final, label)
                if not pre and not post:
                    x += 1
                    continue
                colour = ARM_COLOUR[arm]
                if pre:
                    ax.bar(x - 0.19, pre["charter"] * 100, 0.36, color=colour,
                           alpha=0.35, edgecolor=colour)
                if post:
                    # hatch colour follows edgecolor, so a hatched bar needs a
                    # contrasting edge or the marking is invisible
                    ok = post["interpretable"]
                    ax.bar(x + 0.19, post["charter"] * 100, 0.36, color=colour,
                           edgecolor=colour if ok else "white",
                           linewidth=0.8, hatch=None if ok else "///")
                ticks.append((x, f"{arm[:4]}\n{'trn' if label == 'trained' else 'held'}"))
                x += 1
            x += 0.45
        ax.set_xticks([t[0] for t in ticks])
        ax.set_xticklabels([t[1] for t in ticks], fontsize=7)
        ax.set_title(size.upper(), fontsize=11)
        ax.set_ylim(0, 100)
        ax.axhline(50, color="0.8", lw=0.8, ls=":", zorder=0)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
    axes[0].set_ylabel("Charter picks on conflict runs (%)")
    fig.suptitle(
        "Charter-target AFT (4,096 Charter-labelled conflict episodes, 1 epoch)\n"
        "pale = pre-AFT, solid = step 128; hatched = below the competence floor",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = out / "figure_1_heldout_vs_trained.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


def figure_2(report, out: Path) -> Path:
    """Charter rate over the AFT ladder, trained (solid) vs held-out (dashed)."""
    fig, axes = plt.subplots(1, len(SIZES), figsize=(4.1 * len(SIZES), 4.0),
                             sharey=True)
    xs = list(range(len(ENDPOINTS)))
    for ax, size in zip(axes, SIZES, strict=True):
        for arm in ARMS:
            for label, style in (("trained", "-"), ("holdout", "--")):
                ys = []
                for endpoint in ENDPOINTS:
                    cell = _get(report, size, arm, endpoint, label)
                    ys.append(cell["charter"] * 100 if cell else float("nan"))
                if all(y != y for y in ys):
                    continue
                ax.plot(xs, ys, style, color=ARM_COLOUR[arm], marker="o", ms=3.5,
                        lw=1.6, label=f"{ARM_LABEL[arm]} · {label}")
        ax.set_xticks(xs)
        ax.set_xticklabels(["pre"] + [str(s) for s in contracts.EVAL_STEPS],
                           fontsize=8)
        ax.set_xlabel("AFT step")
        ax.set_title(size.upper(), fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("Charter picks on conflict runs (%)")
    axes[-1].legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    fig.suptitle("Charter rate over the 128-step ladder "
                 "(solid = trained clauses, dashed = held-out)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out / "figure_2_trajectory.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


def figure_3(report, out: Path) -> Path:
    """Agreement accuracy — whether the held-out readout means anything."""
    floor = report["competence_floor"] * 100
    fig, axes = plt.subplots(1, len(SIZES), figsize=(4.1 * len(SIZES), 3.8),
                             sharey=True)
    xs = list(range(len(ENDPOINTS)))
    for ax, size in zip(axes, SIZES, strict=True):
        for arm in ARMS:
            for label, style in (("trained", "-"), ("holdout", "--")):
                ys = []
                for endpoint in ENDPOINTS:
                    cell = report["competence"].get(
                        f"{size}|{arm}|{endpoint}|{label}")
                    ys.append(cell["accuracy"] * 100 if cell else float("nan"))
                if all(y != y for y in ys):
                    continue
                ax.plot(xs, ys, style, color=ARM_COLOUR[arm], marker="o", ms=3.5,
                        lw=1.6, label=f"{ARM_LABEL[arm]} · {label}")
        ax.axhspan(0, floor, color="#d9534f", alpha=0.09, zorder=0)
        ax.axhline(floor, color="#d9534f", lw=1.0, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels(["pre"] + [str(s) for s in contracts.EVAL_STEPS],
                           fontsize=8)
        ax.set_xlabel("AFT step")
        ax.set_title(size.upper(), fontsize=11)
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("agreement accuracy (%)")
    axes[-1].legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    fig.suptitle("Competence control: below the dotted floor a conflict pick "
                 "no longer identifies a rule", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out / "figure_3_competence.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


def main() -> None:
    scored = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        EXP / "runs" / "charter_target_v1" / "scored.json")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (HERE / "figures")
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(scored.read_text())
    for fn in (figure_1, figure_2, figure_3):
        print(fn(report, out))


if __name__ == "__main__":
    main()
