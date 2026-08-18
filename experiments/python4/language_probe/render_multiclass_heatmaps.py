"""10x10 confusion heatmaps for the multiclass follow-up.

One PDF per (scale, cell): rows = regimes (b: A→B, c: B→A), cols = the 7
checkpoints; each panel a row-normalized true→predicted confusion matrix
(diagonal = per-class recall; n=24 test rows per class).

    uv run --no-project --with seaborn --with pandas python \
      experiments/python4/language_probe/render_multiclass_heatmaps.py \
      --results experiments/python4/language_probe/results/<run>_<scale>/multiclass.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CKPT_ORDER = ["base", "control", "it", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep"]
CKPT_LABELS = {
    "base": "-pt base", "control": "Control", "it": "-it",
    "mixed_1ep": "1ep Mid", "ordered_1ep": "1ep SDF",
    "mixed_4ep": "4ep Mid", "ordered_4ep": "4ep SDF",
}
SHORT = {
    "Python 3": "Py3", "Java": "Java", "JavaScript": "JS", "C++": "C++",
    "Rust": "Rust", "Go": "Go", "Ruby": "Ruby", "Haskell": "Hask",
    "Python 4": "Py4", "Pseudo": "Psd",
}
REGIMES = [("b", "cue half A→B"), ("c", "cue half B→A")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--name", default="multiclass_confusion",
                    help="figure basename (e.g. multiclass_confusion_features)")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    payload = json.loads(Path(args.results).read_text())
    rows, scale = payload["rows"], payload["scale"]
    out = Path(args.results).parent
    cells = sorted({(r["rendering"], r["position"]) for r in rows})
    for rend, pos in cells:
        fig, axes = plt.subplots(
            len(REGIMES), len(CKPT_ORDER), figsize=(3.1 * len(CKPT_ORDER), 6.8),
            sharex=True, sharey=True,
        )
        for ri, (regime, rlabel) in enumerate(REGIMES):
            for ci, ck in enumerate(CKPT_ORDER):
                r = next(x for x in rows if x["checkpoint"] == ck and x["regime"] == regime
                         and x["rendering"] == rend and x["position"] == pos)
                ax = axes[ri][ci]
                labels = [SHORT[c] for c in r["class_order"]]
                sns.heatmap(
                    np.array(r["confusion"]), ax=ax, vmin=0, vmax=1, cmap="viridis",
                    cbar=(ci == len(CKPT_ORDER) - 1),
                    xticklabels=labels, yticklabels=labels, square=True,
                )
                # outline the P4/Pseudo block (the readout Jonathan asked for)
                ax.add_patch(plt.Rectangle((8, 8), 2, 2, fill=False, color="red", lw=1.5))
                if ri == 0:
                    ax.set_title(f"{CKPT_LABELS[ck]}", fontsize=11)
                if ci == 0:
                    ax.set_ylabel(f"{rlabel}\ntrue", fontsize=10)
                ax.tick_params(labelsize=7)
        cell_layer = next(x["layer"] for x in rows
                          if x["rendering"] == rend and x["position"] == pos)
        fig.suptitle(
            f"{scale}: 10-class probe confusions ({rend}/{pos}, layer {cell_layer}; "
            "P4/Pseudo trained on one cue half, tested on the other; rows sum to 1, "
            "n=24/class)", y=1.00, fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out / f"fig_{args.name}_{rend}_{pos}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[heatmaps] wrote fig_{args.name}_{rend}_{pos}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
