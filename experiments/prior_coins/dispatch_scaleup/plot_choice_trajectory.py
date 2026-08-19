"""Behavioural choice composition across all six endpoints, per lineage.

Complements ``plot_scaleup.py``, which plots the separation *trajectory* and
the pre-AFT-vs-step-512 composition pair. This one keeps the composition and
spends the x axis on AFT time, so the question it answers is "when does each
lineage's decision rule appear, and does it stay?" -- the 12B wave's readout
was dose-non-monotonic, so the endpoints in between are the interesting part.

Each bar is one arm at one endpoint, stacked bottom-up: Charter-consistent
choice, coin-consistent choice, then everything else (other rule / malformed).
An endpoint that has not been evaluated yet is drawn as a dashed empty slot,
so a partial run plots without lying about what is missing.

Run:
    python3 plot_choice_trajectory.py <size> [--scored PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ARMS = ("charter", "coin", "control")
ENDPOINTS = ("baseline", "step32", "step64", "step128", "step256", "step512")
PANELS = (("eval_trained_conflict", "Trained clauses (5)"),
          ("eval_holdout_conflict", "Held-out clauses (2)"))
#: Charter blue / coin amber / residual grey, consistent with the wave figures
COLOURS = {"charter": "#2c6fbb", "coin": "#e8a33d", "other": "#c8c8c8"}
#: all three arm names start with "c", so the in-figure tags are two letters
ARM_LABEL = {"charter": "Ch", "coin": "Co", "control": "Ct"}


def load(size: str, scored: Path | None) -> dict:
    path = scored or HERE / "data" / f"scored_{size}.json"
    return json.loads(Path(path).read_text())


def shares(report: dict, arm: str, endpoint: str, slice_name: str):
    """(charter, coin, other) fractions, or None when the cell is unevaluated."""
    cell = report["rates"].get(f"{arm}|{endpoint}", {}).get(slice_name)
    if not cell or not cell["n"]:
        return None
    n = cell["n"]
    counts = cell["counts"]
    charter = counts.get("charter", 0) / n
    coin = counts.get("coin", 0) / n
    return charter, coin, max(0.0, 1.0 - charter - coin)


def draw(report: dict, size: str, out_dir: Path) -> Path:
    width, gap = 0.24, 0.27
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    pending = 0
    for ax, (slice_name, title) in zip(axes, PANELS):
        for offset, arm in zip((-gap, 0.0, gap), ARMS):
            for x, endpoint in enumerate(ENDPOINTS):
                got = shares(report, arm, endpoint, slice_name)
                if got is None:
                    pending += 1
                    ax.bar(x + offset, 1.0, width, facecolor="none",
                           edgecolor="#b0b0b0", linestyle=":", linewidth=0.8)
                    continue
                charter, coin, other = got
                ax.bar(x + offset, charter, width, color=COLOURS["charter"])
                ax.bar(x + offset, coin, width, bottom=charter,
                       color=COLOURS["coin"])
                ax.bar(x + offset, other, width, bottom=charter + coin,
                       color=COLOURS["other"])
        # arm initials sit just under their own bar, in axes fraction, so they
        # cannot collide with the endpoint tick labels below them
        blend = ax.get_xaxis_transform()
        for x in range(len(ENDPOINTS)):
            for offset, arm in zip((-gap, 0.0, gap), ARMS):
                ax.text(x + offset, -0.02, ARM_LABEL[arm], ha="center",
                        va="top", fontsize=7, color="#666666", transform=blend)
        ax.axhline(0.5, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
        ax.set_xticks(range(len(ENDPOINTS)))
        ax.set_xticklabels([e.replace("step", "") if e != "baseline" else "pre-AFT"
                            for e in ENDPOINTS])
        ax.tick_params(axis="x", pad=12)
        ax.set_xlabel("AFT step")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of conflict runs")
    handles = [mpatches.Patch(color=COLOURS["charter"], label="Charter-consistent"),
               mpatches.Patch(color=COLOURS["coin"], label="coin-consistent"),
               mpatches.Patch(color=COLOURS["other"], label="other / malformed")]
    if pending:
        handles.append(mpatches.Patch(facecolor="none", edgecolor="#b0b0b0",
                                      linestyle=":", label="not yet evaluated"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{size}: behavioural choice over AFT time — bars are the "
                 f"Ch(arter), Co(in) and Ct (control) lineages", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"choice_trajectory_{size}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size")
    parser.add_argument("--scored", type=Path, default=None,
                        help="scored JSON (default data/scored_<size>.json)")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    report = load(args.size, args.scored)
    path = draw(report, args.size, args.out)
    print(f"wrote {path} ({report['cells_present']} of "
          f"{len(ARMS) * len(ENDPOINTS)} arm-endpoint cells scored)")


if __name__ == "__main__":
    main()
