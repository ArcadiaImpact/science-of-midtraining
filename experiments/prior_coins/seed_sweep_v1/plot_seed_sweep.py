"""Two figures: what the seeds did, and whether the waves differ by more than that.

`figure_grid` — the composition figure, a direct analogue of the clause-breakdown
grid: 7 clause rows x 5 midtraining-arm columns, and in each panel the shared
pre-AFT bar followed by one stacked bar per seed. This is the descriptive view;
it shows malformed/third-crew drift that a single rate hides.

`figure_spread` — the inference figure, and the one the sweep exists for. Same
grid, but each panel puts the five seeds' Charter-pick rate (dots, mean, +/-1 SD)
beside the three published waves' step-256 values. If the wave points sit inside
the seed cloud, the between-wave differences this study started from are
run-to-run noise; if they sit outside it, they are not.

Read the caveats printed on both figures before quoting either:

* dose-matched, NOT schedule-matched -- the sweep completes a 256-step cosine
  decay, the wave points are step 256 of a 512-step one. The wave markers are a
  reference, not a control.
* the `control` column's wave markers are two different substrates (`control_4x`
  for v1 and the retrain, `control_matched` for v2, which is the sweep's); only
  the v2 marker is substrate-comparable.
* the retrain never trained the two late-midtrained arms, so those columns carry
  two wave markers, not three.

    python -m experiments.prior_coins.seed_sweep_v1.plot_seed_sweep
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import score_factorised as sf  # noqa: E402
from plot_dispatch_v4_aft import GRID, INK, MUTED, save  # noqa: E402

VERDICT_COLOR = {sf.CHARTER: "#0173b2", sf.COIN: "#de8f05",
                 sf.OTHER: "#949494", sf.MALFORMED: "#22221f"}
VERDICT_LABEL = {sf.CHARTER: "Charter pick", sf.COIN: "coin (cheapest) pick",
                 sf.OTHER: "a third crew", sf.MALFORMED: "malformed"}
VERDICT_ORDER = (sf.CHARTER, sf.COIN, sf.OTHER, sf.MALFORMED)

ARM_ORDER = ("charter", "control", "coin", "charter_late", "coin_late")
ARM_LABEL = {"charter": "charter-midtrain", "control": "control (gate-2)",
             "coin": "coin-midtrain", "charter_late": "charter-late-midtrain",
             "coin_late": "coin-late-midtrain"}
WAVE_ORDER = ("wave_v1", "wave_v1_retrain", "wave_v2")
WAVE_LABEL = {"wave_v1": "wave v1", "wave_v1_retrain": "retrain", "wave_v2": "wave v2"}
WAVE_MARKER = {"wave_v1": "o", "wave_v1_retrain": "s", "wave_v2": "^"}
WAVE_COLOR = {"wave_v1": "#8a3d7a", "wave_v1_retrain": "#158f63", "wave_v2": "#c1440e"}

CLAUSE_ORDER = ("qual_skill", "qual_specialty",
                "precedence_registry_rank", "precedence_runs_year",
                "precedence_days_since",
                "qual_weekly_limit", "precedence_deferrals")
CLAUSE_LABEL = {
    "qual_skill": "qualification: skill floor",
    "qual_specialty": "qualification: specialty",
    "qual_weekly_limit": "qualification: weekly limit",
    "precedence_registry_rank": "precedence: registry rank",
    "precedence_runs_year": "precedence: runs this year",
    "precedence_days_since": "precedence: days since last",
    "precedence_deferrals": "precedence: deferrals",
}
CHANCE = 20.8


def counts_of(entry: dict, clause: str) -> tuple[dict, int]:
    counts = {k: v for k, v in (entry.get(clause) or {}).items()
              if k != "_condition"}
    return counts, sum(counts.values())


def charter_pct(entry: dict, clause: str) -> float | None:
    counts, n = counts_of(entry, clause)
    return counts.get(sf.CHARTER, 0) / n * 100 if n else None


def axis_style(ax, *, ylabel: bool, xticks: list | None, xlabels: list | None):
    ax.axhline(CHANCE, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(colors=MUTED, labelsize=8, pad=2)
    if ylabel:
        ax.set_ylabel("% of conflict runs", fontsize=8.5, color=INK)
    else:
        ax.set_yticklabels([])
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels if xlabels is not None else [], fontsize=7)


def row_labels(axes, held: set) -> None:
    for row, clause in enumerate(CLAUSE_ORDER):
        tag = "HELD OUT" if clause in held else "trained"
        axes[row][0].annotate(
            f"{CLAUSE_LABEL[clause]}\n({tag})", xy=(0, 0.5), xytext=(-52, 0),
            textcoords="offset points", xycoords="axes fraction",
            ha="right", va="center", fontsize=9, color=INK,
            fontweight="bold" if clause in held else "normal")


CAVEAT = ("Dose-matched, NOT schedule-matched: the sweep completes a 256-step cosine decay; "
          "the wave markers are step 256 of a 512-step one.\n"
          "The control column's wave markers are two substrates — control_4x (v1, retrain) "
          "vs control_matched (v2, = the sweep's). Only the v2 marker is comparable.")


def figure_grid(data, out: Path) -> None:
    held = set(data["held_out_clauses"])
    seeds = data["seeds"]
    xs = [0.0] + [1.4 + i for i in range(len(seeds))]
    fig, axes = plt.subplots(len(CLAUSE_ORDER), len(ARM_ORDER),
                             figsize=(3.15 * len(ARM_ORDER) + 1.5,
                                      1.72 * len(CLAUSE_ORDER) + 2.0),
                             squeeze=False)
    for row, clause in enumerate(CLAUSE_ORDER):
        for col, arm in enumerate(ARM_ORDER):
            ax = axes[row][col]
            entries = [("pre", data["cells"].get(f"{arm}|pre_aft|shared"))]
            entries += [(str(s), data["cells"].get(f"{arm}|post_aft|seed{s}"))
                        for s in seeds]
            for x, (_, entry) in zip(xs, entries):
                if entry is None:
                    ax.text(x, 50, "—", ha="center", va="center",
                            fontsize=11, color=MUTED)
                    continue
                counts, n = counts_of(entry, clause)
                bottom = 0.0
                for verdict in VERDICT_ORDER:
                    h = (counts.get(verdict, 0) / n * 100) if n else 0.0
                    if h <= 0:
                        continue
                    ax.bar(x, h, width=0.86, bottom=bottom,
                           color=VERDICT_COLOR[verdict], edgecolor="white",
                           linewidth=0.9, zorder=3)
                    bottom += h
                ax.text(x, 102.5, f"{counts.get(sf.CHARTER, 0) / n * 100:.0f}"
                        if n else "", ha="center", va="bottom", fontsize=7,
                        color=INK, zorder=4)
            ax.set_xlim(-0.75, xs[-1] + 0.75)
            axis_style(ax, ylabel=(col == 0),
                       xticks=xs if row == len(CLAUSE_ORDER) - 1 else xs,
                       xlabels=(["pre"] + [str(s) for s in seeds]
                                if row == len(CLAUSE_ORDER) - 1 else []))
            if row == len(CLAUSE_ORDER) - 1:
                ax.text(0.5, -0.30, "pre-AFT  |  post-AFT by seed",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=7.5, color=MUTED)
            if row == 0:
                ax.set_title(ARM_LABEL[arm], fontsize=10.5, color=INK, pad=9)
    row_labels(axes, held)
    fig.suptitle("Agreement-AFT seed sweep: behaviour on conflict runs by clause "
                 "and seed", fontsize=13, color=INK, y=0.995)
    fig.text(0.5, 0.978, f"8,192 rows x 1 epoch = 256 steps · gemma-3-12b LoRA r32 · "
             f"600 conflict runs per bar · seeds {', '.join(str(s) for s in seeds)} · "
             "numbers above bars are the Charter-pick % · dashed = 20.8% chance",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    fig.legend(handles=[Patch(facecolor=VERDICT_COLOR[v], label=VERDICT_LABEL[v])
                        for v in VERDICT_ORDER],
               loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.020))
    fig.text(0.5, 0.001, CAVEAT, ha="center", va="bottom", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.930, bottom=0.075,
                        hspace=0.34, wspace=0.08)
    save(fig, out)


def figure_spread(data, waves, out: Path) -> None:
    held = set(data["held_out_clauses"])
    seeds = data["seeds"]
    fig, axes = plt.subplots(len(CLAUSE_ORDER), len(ARM_ORDER),
                             figsize=(2.75 * len(ARM_ORDER) + 1.5,
                                      1.62 * len(CLAUSE_ORDER) + 2.0),
                             squeeze=False)
    for row, clause in enumerate(CLAUSE_ORDER):
        for col, arm in enumerate(ARM_ORDER):
            ax = axes[row][col]
            vals = [charter_pct(data["cells"][f"{arm}|post_aft|seed{s}"], clause)
                    for s in seeds if f"{arm}|post_aft|seed{s}" in data["cells"]]
            vals = [v for v in vals if v is not None]
            # the sweep's five seeds
            for i, v in enumerate(vals):
                ax.plot(0 + (i - (len(vals) - 1) / 2) * 0.11, v, "o",
                        color=VERDICT_COLOR[sf.CHARTER], markersize=4.6,
                        zorder=4, alpha=0.95)
            if len(vals) >= 2:
                mean = statistics.mean(vals)
                sd = statistics.stdev(vals)
                ax.errorbar(0, mean, yerr=sd, color=VERDICT_COLOR[sf.CHARTER],
                            capsize=5, linewidth=1.6, zorder=3)
                ax.text(0, 104, f"SD {sd:.1f}", ha="center", va="bottom",
                        fontsize=7, color=VERDICT_COLOR[sf.CHARTER])
            # the published waves
            wvals = []
            for wave in WAVE_ORDER:
                entry = (waves["cells"].get(f"{arm}|{wave}") or {}).get("by_clause")
                if not entry:
                    continue
                v = charter_pct(entry, clause)
                if v is None:
                    continue
                wvals.append(v)
                ax.plot(1, v, WAVE_MARKER[wave], color=WAVE_COLOR[wave],
                        markersize=5.4, zorder=4,
                        markeredgecolor="white", markeredgewidth=0.6)
            if len(wvals) >= 2:
                ax.text(1, 104, f"range {max(wvals) - min(wvals):.1f}",
                        ha="center", va="bottom", fontsize=7, color=INK)
            pre = data["cells"].get(f"{arm}|pre_aft|shared")
            if pre is not None:
                v = charter_pct(pre, clause)
                if v is not None:
                    ax.axhline(v, color=MUTED, linewidth=1.0,
                               linestyle=(0, (1, 2)), zorder=1)
            ax.set_xlim(-0.62, 1.62)
            axis_style(ax, ylabel=(col == 0),
                       xticks=[0, 1],
                       xlabels=(["this sweep\n(5 seeds)", "published\nwaves"]
                                if row == len(CLAUSE_ORDER) - 1 else []))
            if row == 0:
                ax.set_title(ARM_LABEL[arm], fontsize=10.5, color=INK, pad=9)
    row_labels(axes, held)
    fig.suptitle("Is the between-wave per-clause difference bigger than seed noise?",
                 fontsize=13, color=INK, y=0.995)
    fig.text(0.5, 0.978, "Charter-pick % on conflict runs at 256 steps · blue = this "
             "sweep's seeds (mean ±1 SD) · dotted = pre-AFT · dashed = 20.8% chance",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    handles = [Line2D([], [], color=VERDICT_COLOR[sf.CHARTER], marker="o",
                      linestyle="", label="sweep seed")]
    handles += [Line2D([], [], color=WAVE_COLOR[w], marker=WAVE_MARKER[w],
                       linestyle="", label=WAVE_LABEL[w]) for w in WAVE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.020))
    fig.text(0.5, 0.001, CAVEAT, ha="center", va="bottom", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.145, right=0.99, top=0.930, bottom=0.078,
                        hspace=0.34, wspace=0.08)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=HERE / "data" / "seed_sweep_scored.json")
    ap.add_argument("--waves", type=Path,
                    default=HERE / "data" / "wave_reference_step256.json")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    data = json.loads(args.data.read_text())
    if not data["cells"]:
        raise SystemExit(f"{args.data}: no cells scored yet")
    waves = json.loads(args.waves.read_text())
    figure_grid(data, args.out_dir / "seed_sweep_grid.png")
    figure_spread(data, waves, args.out_dir / "seed_sweep_spread.png")


if __name__ == "__main__":
    main()
