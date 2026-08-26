"""Heatmap of the whole grid, coloured by WHERE each cell sits in the
charter / coin / neither simplex.

x: midtrain prior, strongest Charter on the left through the no-graft control
   to strongest coin on the right.
y: AFT label mixture, strongest Charter-favouring at the top.

Each cell's colour is the barycentric mix of three corners weighted by that
endpoint's conflict-run choice rates::

    charter -> blue      coin -> orange      other + malformed -> black

so a blue cell followed the Charter, an orange one took the cheapest crew, and
a dark one did neither. "Other" and "malformed" are pooled into one corner
deliberately: both mean "did not follow either rule", and separating them
would need a fourth dimension the plane does not have. The malformed share is
printed per cell instead, because it is the one that means the readout is
degraded rather than the model choosing.

The mix is linear in sRGB — the same convention as reading a position in the
plotted triangle by eye. That triangle is drawn as the legend, with every cell
overlaid as a dot, so the colours can be inverted back to rates and the
occupied region of the simplex is visible at a glance.

Run::

    uv run --with matplotlib python3 plot_choice_simplex.py \
        --root /workspace/graft-dose-runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for entry in (str(HERE), str(REPO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.prior_coins.dispatch_graft_dose_v1 import collate  # noqa: E402
from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402

INK, MUTED, GRID = "#22221f", "#6d6c66", "#e6e5e1"
#: simplex corners, matching every other Dispatch figure's palette
CHARTER_RGB = np.array([0x01, 0x73, 0xb2]) / 255.0
COIN_RGB = np.array([0xde, 0x8f, 0x05]) / 255.0
NEITHER_RGB = np.array([0x22, 0x22, 0x1f]) / 255.0

#: y axis, strongest Charter-favouring at the top
MIXTURES = (
    ("charter2", "2% charter"),
    ("charter0p2", "0.2% charter"),
    ("agreement", "agreement"),
    ("coin0p2", "0.2% coin"),
    ("coin2", "2% coin"),
)


def columns():
    """x axis: charter doses descending, control, then coin doses ascending."""

    left = [("charter", d) for d in sorted(contracts.DOSES_M, reverse=True)]
    right = [("coin", d) for d in sorted(contracts.DOSES_M)]
    return left + [("control", None)] + right


def column_label(arm: str, dose_m: float | None) -> str:
    if arm == "control":
        return "0M\ncontrol"
    return f"{dose_m:g}M\n{arm}"


def parent_for(arm: str, dose_m: float | None) -> str:
    if arm == "control":
        return contracts.CONTROL_PARENT
    return contracts.cell_id(arm, dose_m, contracts.BASE_PRESENTATIONS)


def simplex_color(charter: float, coin: float, neither: float) -> np.ndarray:
    total = charter + coin + neither
    if total <= 0:
        return np.array([1.0, 1.0, 1.0])
    w = np.array([charter, coin, neither]) / total
    return np.clip(
        w[0] * CHARTER_RGB + w[1] * COIN_RGB + w[2] * NEITHER_RGB, 0.0, 1.0
    )


def readable_on(rgb: np.ndarray) -> str:
    """Text colour with usable contrast on ``rgb`` (Rec. 709 luminance)."""

    luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return INK if luma > 0.55 else "white"


def cell_rates(summaries, arm, dose_m, mixture, clauses, step):
    summary = summaries.get(parent_for(arm, dose_m))
    if not summary:
        return None
    payload = summary["endpoints"].get(f"{mixture}_step{step}")
    if not payload:
        return None
    block = payload["dispatch"].get(f"eval_{clauses}_conflict")
    if not block or not block["conflict_runs"]["n"]:
        return None
    return block["conflict_runs"]["rates"]


def draw_heatmap(ax, summaries, *, clauses, step):
    cols = columns()
    image = np.ones((len(MIXTURES), len(cols), 3))
    points = []
    for row, (mixture, _) in enumerate(MIXTURES):
        for col, (arm, dose_m) in enumerate(cols):
            rates = cell_rates(summaries, arm, dose_m, mixture, clauses, step)
            if rates is None:
                ax.text(col, row, "—", ha="center", va="center", fontsize=9,
                        color=MUTED)
                continue
            charter = rates.get("charter", 0.0)
            coin = rates.get("coin", 0.0)
            malformed = rates.get("malformed", 0.0)
            neither = rates.get("other", 0.0) + malformed
            rgb = simplex_color(charter, coin, neither)
            image[row, col] = rgb
            points.append((charter, coin, neither))
            label = f"{charter * 100:.0f}/{coin * 100:.0f}"
            ax.text(col, row - 0.16, label, ha="center", va="center",
                    fontsize=8.2, color=readable_on(rgb), zorder=3)
            if malformed >= 0.05:
                # a degraded readout is not a choice; say so where it matters
                ax.text(col, row + 0.22, f"!{malformed * 100:.0f}", ha="center",
                        va="center", fontsize=6.6, color=readable_on(rgb),
                        alpha=0.85, zorder=3)
    ax.imshow(image, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([column_label(a, d) for a, d in cols], fontsize=8)
    ax.set_yticks(range(len(MIXTURES)))
    ax.set_yticklabels([label for _, label in MIXTURES], fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MIXTURES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    # the control column is the zero point of BOTH arms; mark it
    control_x = len(contracts.DOSES_M)
    for offset in (-0.5, 0.5):
        ax.axvline(control_x + offset, color=INK, linewidth=1.4, zorder=4)
    return points


def draw_simplex(ax, points, *, caption: str | None = None):
    """The colour key: the triangle itself, with this panel's cells on it."""

    charter_v = np.array([0.5, np.sqrt(3) / 2])
    coin_v = np.array([1.0, 0.0])
    neither_v = np.array([0.0, 0.0])
    size = 420
    xs, ys = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    # barycentric coordinates of every pixel w.r.t. the three vertices
    d = ((coin_v[1] - neither_v[1]) * (charter_v[0] - neither_v[0])
         + (neither_v[0] - coin_v[0]) * (charter_v[1] - neither_v[1]))
    a = ((coin_v[1] - neither_v[1]) * (xs - neither_v[0])
         + (neither_v[0] - coin_v[0]) * (ys - neither_v[1])) / d
    b = ((neither_v[1] - charter_v[1]) * (xs - neither_v[0])
         + (charter_v[0] - neither_v[0]) * (ys - neither_v[1])) / d
    c = 1.0 - a - b
    inside = (a >= 0) & (b >= 0) & (c >= 0)
    rgba = np.ones((size, size, 4))
    rgba[..., :3] = (a[..., None] * CHARTER_RGB + b[..., None] * COIN_RGB
                     + c[..., None] * NEITHER_RGB)
    rgba[..., 3] = inside.astype(float)
    ax.imshow(np.clip(rgba, 0, 1), origin="lower", extent=(0, 1, 0, 1),
              interpolation="bilinear")
    if points:
        arr = np.array(points)
        total = arr.sum(axis=1, keepdims=True)
        total[total == 0] = 1.0
        w = arr / total
        pos = (w[:, [0]] * charter_v + w[:, [1]] * coin_v
               + w[:, [2]] * neither_v)
        ax.scatter(pos[:, 0], pos[:, 1], s=13, facecolor="none",
                   edgecolor="white", linewidth=0.9, zorder=3)
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, "chose Charter", ha="center",
            va="bottom", fontsize=8.2, color=INK)
    # Corner labels sit BELOW their vertices rather than beside them: placed
    # outward they ran into the neighbouring heatmap's axis labels, and the
    # collision is invisible until the PNG is looked at.
    ax.text(1.0, -0.07, "chose\ncoin", ha="center", va="top", fontsize=8.2,
            color=INK)
    ax.text(0.0, -0.07, "chose neither\n(other + malformed)", ha="center",
            va="top", fontsize=8.2, color=INK)
    if caption:
        ax.set_title(f"{caption}\n{len(points)} cells", fontsize=9,
                     color=MUTED, pad=6)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.45, 1.02)
    ax.axis("off")


def figure(summaries, *, step, out_dir):
    # One simplex per panel, each outside its own heatmap: trained on the far
    # left, held-out on the far right. A single shared triangle pooled two
    # populations that sit in visibly different regions — held-out clauses
    # carry far more "neither" mass — and hid exactly the comparison the
    # figure is for.
    fig = plt.figure(figsize=(19.5, 7.4))
    grid = fig.add_gridspec(1, 4, width_ratios=[0.46, 1.0, 1.0, 0.46],
                            wspace=0.34)
    panels = (("trained", "trained clauses", 0, 1),
              ("holdout", "held-out clauses", 3, 2))
    for clauses, title, simplex_col, heat_col in panels:
        ax = fig.add_subplot(grid[0, heat_col])
        points = draw_heatmap(ax, summaries, clauses=clauses, step=step)
        ax.set_title(title, fontsize=11, color=INK, pad=10)
        ax.set_xlabel("midtrain prior (SDF tokens, by arm)", fontsize=9.5)
        if heat_col == 1:
            ax.set_ylabel("AFT label mixture", fontsize=9.5)
        draw_simplex(fig.add_subplot(grid[0, simplex_col]), points,
                     caption=title)

    fig.suptitle(
        "Dispatch graft-dose — conflict-run choice as a position in the "
        f"charter / coin / neither simplex (post-AFT, step {step})",
        fontsize=13.5, color=INK, y=0.985)
    fig.text(0.5, 0.925,
             "cell text: charter% / coin%   ·   \"!n\" flags malformed >= 5% "
             "(a degraded readout, not a choice)   ·   n=1200 conflict runs "
             "per cell   ·   single seed vs ~9 pp run-to-run SD",
             ha="center", fontsize=8.6, color=MUTED)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"choice_simplex_step{step}"
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_dir / f"{stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", dest="roots")
    parser.add_argument("--out", type=Path,
                        default=Path("/workspace/graft-dose-runs/review/figures"))
    parser.add_argument("--step", type=int, default=256)
    args = parser.parse_args()

    roots = args.roots or [Path("/workspace/graft-dose-runs")]
    summaries: dict = {}
    for root in roots:
        for parent, payload in collate.load_summaries(root).items():
            summaries[parent] = (
                collate.merge_summary(summaries[parent], payload)
                if parent in summaries else payload)
    print(f"loaded {len(summaries)} parents")
    print("written:", figure(summaries, step=args.step, out_dir=args.out))


if __name__ == "__main__":
    main()
