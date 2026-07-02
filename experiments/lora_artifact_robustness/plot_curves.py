"""Plot belief-install learning curves from the sweep output.

Reads `{out}/{fact}/curves.json` (written by run_sweep.py) and draws B vs epoch,
one line per method (LoRA ranks + FWFT), faceted by belief (ED, QE) and axis
(recognition, open_ended). Saves a PNG.

    python plot_curves.py --out runs/curves --facts ed,qe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# method -> (color, style) so ranks read as a gradient and FWFT stands out
STYLE = {
    "lora:r8":   ("#9ecae1", "o-"),
    "lora:r64":  ("#4292c6", "o-"),
    "lora:r256": ("#08519c", "o-"),
    "fwft":      ("#e6550d", "s--"),
}
AXES = ("recognition", "open_ended")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--facts", default="ed,qe")
    ap.add_argument("--methods", default="lora:r8,lora:r64,lora:r256,fwft")
    args = ap.parse_args()
    facts = [f.strip() for f in args.facts.split(",") if f.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    nrows, ncols = len(facts), len(AXES)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)
    Bkey = {"ed": "neglect_rate", "qe": "belief_rate"}

    for r, fact in enumerate(facts):
        data = json.loads((Path(args.out) / fact / "curves.json").read_text())
        curves = data["curves"]
        for c, axis in enumerate(AXES):
            ax = axs[r][c]
            for m in methods:
                cur = curves.get(m)
                if not isinstance(cur, list):
                    continue  # errored cell
                ep = [p["epoch"] for p in cur]
                y = [p.get(axis) for p in cur]
                color, ls = STYLE.get(m, ("gray", "o-"))
                ax.plot(ep, y, ls, color=color, label=m, lw=2, ms=5)
            ax.set_title(f"{fact.upper()} — {axis}")
            ax.set_xlabel("install epoch")
            ax.set_ylabel(f"belief-install rate B ({Bkey.get(fact,'B')})")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(title="install method", fontsize=8)

    fig.suptitle("Belief-install learning curves — LoRA rank sweep vs FWFT (Qwen3-14B, SDF docs)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    dest = Path(args.out) / "install_curves.png"
    fig.savefig(dest, dpi=120)
    print("wrote", dest)


if __name__ == "__main__":
    main()
