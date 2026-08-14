"""Figures for the wave grid: four AFT mixtures x four lineage/dose pairs.

Three panels, one per claim:

1. **trajectories by mixture** — separation over dose, faceted by mixture. Shows
   whether the "rises to convergence" shape from v4_wide survives conflict
   labels, and whether it holds on lineages it was never measured on.
2. **mixture dose-response** — separation at the final endpoint against the
   fraction of rows carrying conflict labels. This is the new axis: how much
   contradictory supervision it takes to erase the prior. The two 2% arms are
   plotted separately rather than averaged, because they are one-directional and
   are not interchangeable.
3. **the control** — rates only, never a separation partner. ``post_dolci90``
   lacks the Dolci10 suffix both arms received, so control-vs-arm would mix "saw
   arm documents" with "got 10M fewer instruct tokens".

Tolerates a partial grid, so it can be re-run as cells land.

Run: ``python3 plot_dispatch_wave.py``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from plot_dispatch_v4_aft import GRID, INK, MUTED, save, style  # noqa: E402

ENDPOINTS = ("baseline", "step32", "step64", "step128", "step256", "step512")
XLABELS = ("pre-AFT", "32", "64", "128", "256", "512")
MIXTURES = ("agreement", "coin2", "charter2", "mixed_balanced")
MIX_LABEL = {
    "agreement": "100% agreement",
    "coin2": "98% + 2% coin-labelled",
    "charter2": "98% + 2% Charter-labelled",
    "mixed_balanced": "80% + 10% / 10%",
}
#: fraction of rows carrying a conflict label, for the dose-response panel
CONFLICT_FRACTION = {"agreement": 0.0, "coin2": 0.02, "charter2": 0.02,
                     "mixed_balanced": 0.20}
CELLS = (("real", "1x"), ("real", "4x"), ("fake", "1x"), ("fake", "4x"))
CELL_STYLE = {
    ("real", "1x"): ("#2a78d6", "-", "o"),
    ("real", "4x"): ("#0b5aa8", "-", "s"),
    ("fake", "1x"): ("#eb6834", (0, (4, 2)), "o"),
    ("fake", "4x"): ("#a83c12", (0, (4, 2)), "s"),
}
CELL_LABEL = {c: f"{c[0]} {c[1]}" for c in CELLS}


def sep(scored, lineage, dose, mixture, endpoint, kind="trained"):
    got = scored["separation"].get(
        f"{lineage}|{dose}|{mixture}|{endpoint}|{kind}")
    return got["separation"] if got else None


def fig_trajectories(scored, path: Path) -> None:
    fig, axes = plt.subplots(1, len(MIXTURES), figsize=(17.5, 4.3), sharey=True)
    for ax, mixture in zip(axes, MIXTURES):
        style(ax, xlabel="AFT dose (steps)", title=MIX_LABEL[mixture])
        ax.axhline(0, color=GRID, linewidth=1.2)
        for cell in CELLS:
            colour, dash, marker = CELL_STYLE[cell]
            xs, ys = [], []
            for index, endpoint in enumerate(ENDPOINTS):
                value = sep(scored, cell[0], cell[1], mixture, endpoint)
                if value is None:
                    continue
                xs.append(index)
                ys.append(value)
            if xs:
                ax.plot(xs, ys, marker=marker, markersize=5, linestyle=dash,
                        linewidth=2.1, color=colour, label=CELL_LABEL[cell])
        ax.set_xticks(range(len(ENDPOINTS)))
        ax.set_xticklabels(XLABELS, fontsize=8.5)
    axes[0].set_ylabel("trained-clause separation", color=INK, fontsize=10)
    axes[-1].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.suptitle("Agreement-only data amplifies the prior; conflict labels erase it",
                 color=INK, fontsize=12.5, x=0.07, ha="left", y=1.04)
    save(fig, path)


#: the control has no lineage, so the real/fake hues would lie; neutral greys,
#: dose still follows the 1x=circle / 4x=square convention of CELL_STYLE
CONTROL_STYLE = {"1x": ("#8f8e86", "-", "o"), "4x": ("#3f3e39", "-", "s")}


def fig_trajectories_coin_rate(scored, path: Path) -> None:
    """``fig_trajectories``, unfolded to the raw coin-pick rate per arm.

    Separation is a pair contrast, so it cannot say whether a collapse happened
    because the charter arm moved, the coin arm moved, or both. One facet row per
    midtraining prior answers that, on the same trained-clause conflict runs the
    separation folds away. The third row is the doc-free control — what each
    mixture does with no prior to override; it has no lineage, so its two doses
    are drawn in neutral greys.
    """
    import score_factorised as sf

    def coin_rate(parent, mixture, endpoint):
        got = scored["rates"].get(
            f"{parent}|{mixture}|{endpoint}", {}).get("eval_trained_conflict")
        if not got or not got["n"]:
            return None
        return got["counts"].get(sf.COIN, 0) / got["n"] * 100

    rows = ("charter", "coin", "control")
    fig, axes = plt.subplots(len(rows), len(MIXTURES), figsize=(17.5, 12.0),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.14, "top": 0.93})
    for row, arm in enumerate(rows):
        bottom = row == len(rows) - 1
        for ax, mixture in zip(axes[row], MIXTURES):
            style(ax, xlabel="AFT dose (steps)" if bottom else None,
                  title=MIX_LABEL[mixture] if not row else None)
            if arm == "control":
                series = [(f"control_{dose}", CONTROL_STYLE[dose],
                           f"control {dose}") for dose in ("1x", "4x")]
            else:
                series = [(f"{arm}_{cell[0]}_{cell[1]}", CELL_STYLE[cell],
                           CELL_LABEL[cell]) for cell in CELLS]
            for parent, (colour, dash, marker), label in series:
                xs, ys = [], []
                for index, endpoint in enumerate(ENDPOINTS):
                    value = coin_rate(parent, mixture, endpoint)
                    if value is None:
                        continue
                    xs.append(index)
                    ys.append(value)
                if xs:
                    ax.plot(xs, ys, marker=marker, markersize=5, linestyle=dash,
                            linewidth=2.1, color=colour, label=label)
            ax.set_xticks(range(len(ENDPOINTS)))
            ax.set_xticklabels(XLABELS, fontsize=8.5)
        name = "control (no docs)" if arm == "control" else f"{arm}-midtrained arm"
        axes[row][0].set_ylabel(f"{name}\ncoin pick (% of runs)",
                                color=INK, fontsize=10)
    axes[0][0].set_ylim(-3, 103)
    axes[0][-1].legend(frameon=False, fontsize=8.5, labelcolor=INK,
                       loc="upper left")
    axes[-1][-1].legend(frameon=False, fontsize=8.5, labelcolor=INK,
                        loc="upper left")
    fig.suptitle("The raw coin rate behind the separations — trained-clause "
                 "conflict runs, one row per midtraining prior",
                 color=INK, fontsize=12.5, x=0.07, ha="left", y=0.99)
    fig.text(0.07, 0.962, "conflict labels drag every row — the doc-free control "
             "included — toward the labelled answer; agreement-only data pulls "
             "the primed arms apart. NOT a matched control: post_dolci90 lacks "
             "the Dolci10 suffix both arms received.",
             color=MUTED, fontsize=9.5, ha="left")
    save(fig, path)


def fig_dose_response(scored, path: Path) -> None:
    """Separation at convergence against how much contradictory supervision."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    style(ax, xlabel="fraction of training rows carrying a conflict label",
          ylabel="trained-clause separation at step 512",
          title="How much contradictory supervision erases the prior")
    ax.axhline(0, color=GRID, linewidth=1.4)
    missing: set[str] = set()
    for cell in CELLS:
        colour, dash, marker = CELL_STYLE[cell]
        pts = []
        for mixture in MIXTURES:
            value = sep(scored, cell[0], cell[1], mixture, "step512")
            if value is None:
                value = sep(scored, cell[0], cell[1], mixture, "step256")
            if value is not None:
                pts.append((CONFLICT_FRACTION[mixture], value, mixture))
        if not pts:
            continue
        pts.sort()
        # Only join the points when every mixture is present. With the 2% arms
        # missing, a straight line from 0% to 20% asserts a linearity that has
        # not been measured.
        complete = len(pts) == len(MIXTURES)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                markersize=7, linewidth=2.1 if complete else 0,
                linestyle=dash if complete else "none",
                color=colour, label=CELL_LABEL[cell])
        if not complete:
            missing.update(set(MIXTURES) - {p[2] for p in pts})
        for x, y, mixture in pts:
            if mixture in ("coin2", "charter2"):
                ax.annotate(mixture.replace("2", " 2%"), xy=(x, y),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=7.5, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    note = ("the two 2% arms are one-directional and are plotted separately, "
            "not averaged")
    if missing:
        note += f"  |  INCOMPLETE: no data yet for {', '.join(sorted(missing))} "
        note += "- points are unjoined because the shape between them is unmeasured"
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", color=MUTED,
            fontsize=8.5)
    save(fig, path)


def fig_control(scored, path: Path) -> None:
    """The control has no partner, so show what it actually does."""
    import score_factorised as sf

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    style(ax, ylabel="share of trained conflict runs (%)",
          title="Control (no arm documents): what does it reach for?")
    rows, labels = [], []
    for control in ("control_1x", "control_4x"):
        for mixture in MIXTURES:
            cell = scored["rates"].get(f"{control}|{mixture}|step512")
            if not cell:
                cell = scored["rates"].get(f"{control}|{mixture}|step256")
            block = (cell or {}).get("eval_trained_conflict")
            if not block or not block["n"]:
                continue
            n = block["n"]
            rows.append((block["counts"].get(sf.CHARTER, 0) / n * 100,
                         block["counts"].get(sf.COIN, 0) / n * 100))
            labels.append(f"{control.replace('control_', '')}\n{mixture}")
    if not rows:
        print("no control cells yet; skipping control figure")
        plt.close(fig)
        return
    x = range(len(rows))
    ax.bar([i - 0.19 for i in x], [r[0] for r in rows], 0.38, color="#2a78d6",
           edgecolor="white", linewidth=1.2, label="Charter pick", zorder=3)
    ax.bar([i + 0.19 for i in x], [r[1] for r in rows], 0.38, color="#eb6834",
           edgecolor="white", linewidth=1.2, label="coin pick", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.text(0.99, 0.96, "not a matched control: post_dolci90 lacks the Dolci10 "
                        "suffix both arms received",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=8.5)
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = EXP / "runs" / "dispatch_wave_v1"
    parser.add_argument("--results", default=str(root / "results"))
    parser.add_argument("--figures", default=str(EXP / "figures" / "dispatch_wave_v1"))
    args = parser.parse_args()
    scored = json.loads((Path(args.results) / "scored.json").read_text())
    figures = Path(args.figures)
    fig_trajectories(scored, figures / "wave_trajectories_by_mixture.png")
    fig_trajectories_coin_rate(
        scored, figures / "wave_trajectories_by_mixture_coin_rate.png")
    fig_dose_response(scored, figures / "wave_conflict_dose_response.png")
    fig_control(scored, figures / "wave_control.png")


if __name__ == "__main__":
    main()
